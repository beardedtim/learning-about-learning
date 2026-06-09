"""
train.py — PPO training loop for the bug world.

Structure:
  1. Config         — all hyperparameters in one place
  2. RolloutBuffer  — stores one rollout's worth of experience, computes returns
  3. train()        — the main loop: rollout → compute returns → PPO update → log
"""

import time
import torch
import torch.nn.functional as F

from world   import World
from species import BugActorCritic


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# All hyperparameters live here. Nothing is scattered through the code.
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Infrastructure ────────────────────────────────────────────────────────
    device          = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir  = "checkpoints"

    # ── World ─────────────────────────────────────────────────────────────────
    num_envs      = 64          # parallel environments
    grid_size     = 24
    min_food      = 100
    food_zones = [
        {"x": 8, "y": 8, "w": 16, "h": 16},  # one big central zone
    ]
    max_life      = 400.0
    food_reward   = 400.0
    life_decay    = 1.0
    # Sensor geometry
    sensor_radius    = 5
    fov_degrees      = 120
    front_fov_radius = 5
    side_fov_radius  = 2

    # ── Bug network ───────────────────────────────────────────────────────────
    hidden_dim    = 64

    # ── PPO rollout ───────────────────────────────────────────────────────────
    # One "update" = collect rollout_steps × num_envs transitions, then run
    # ppo_epochs passes of minibatch gradient descent over that data.
    rollout_steps = 256         # steps per env before each update
    ppo_epochs    = 4           # how many passes over the rollout buffer
    minibatch_size = 512        # samples per gradient step (must divide rollout_steps × num_envs)

    # ── PPO loss coefficients ─────────────────────────────────────────────────
    clip_eps      = 0.2         # PPO clipping range
    value_coeff   = 0.5         # weight of critic loss
    entropy_coeff = 0.01        # weight of entropy bonus (encourages exploration)
    max_grad_norm = 0.5         # gradient clipping

    # ── GAE (Generalised Advantage Estimation) ────────────────────────────────
    # GAE is how we compute advantages from raw rewards.
    # gamma:  discount factor. Rewards far in the future are worth less.
    #         0.99 = the bug cares about the next ~100 steps.
    # gae_lambda: smoothing between pure TD (0.0) and pure Monte Carlo (1.0).
    #         0.95 is the standard starting point.
    gamma      = 0.99
    gae_lambda = 0.95

    # ── Training schedule ─────────────────────────────────────────────────────
    total_steps   = 10_000_000  # total env steps across all envs
    lr            = 3e-4
    log_interval  = 10          # log every N updates
    save_interval = 100         # save checkpoint every N updates


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROLLOUT BUFFER
# Stores everything collected during one rollout, then computes GAE returns.
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Pre-allocates fixed tensors for one rollout and fills them step-by-step.

    WHY PRE-ALLOCATE: Appending to Python lists and stacking at the end is
    fine for small experiments but creates garbage-collection pressure at scale.
    Filling pre-allocated tensors is cleaner and faster.

    WHY GAE: Raw returns (sum of future rewards) have high variance — the bug's
    total reward swings wildly episode to episode. GAE blends single-step TD
    targets (low variance, slightly biased) with full returns (high variance,
    unbiased) via lambda. This gives stable, usable advantage estimates.
    """

    def __init__(self, rollout_steps, num_envs, obs_dim, hidden_dim, device):
        T, E = rollout_steps, num_envs
        self.device = device

        # Transitions collected at each step
        self.obs      = torch.zeros(T, E, obs_dim,    device=device)
        self.actions  = torch.zeros(T, E,             device=device, dtype=torch.long)
        self.log_probs= torch.zeros(T, E,             device=device)
        self.values   = torch.zeros(T, E,             device=device)
        self.rewards  = torch.zeros(T, E,             device=device)
        self.dones    = torch.zeros(T, E,             device=device)

        # Hidden states at each step — needed for evaluate_actions() during update,
        # because the GRU means the same (obs, action) pair can have different
        # log_probs depending on what memory looked like at that moment.
        self.memories = torch.zeros(T, E, hidden_dim, device=device)

        # Filled by compute_returns()
        self.returns   = None
        self.advantages = None

        self.step = 0
        self.T    = T
        self.E    = E

    def store(self, obs, action, log_prob, value, reward, done, memory):
        """Store one timestep of data across all envs."""
        t = self.step
        self.obs[t]       = obs
        self.actions[t]   = action
        self.log_probs[t] = log_prob
        self.values[t]    = value
        self.rewards[t]   = reward
        self.dones[t]     = done.float()
        self.memories[t]  = memory
        self.step += 1

    def compute_returns(self, last_value, last_done):
        """
        Compute GAE advantages and discounted returns in-place.

        last_value: (num_envs,) — V(s) at the step AFTER the rollout ends.
                    Needed to bootstrap the value of the truncated episode.
        last_done:  (num_envs,) bool — whether that final step was terminal.
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae   = torch.zeros(self.E, device=self.device)

        # Walk backwards through time — GAE is a recursive formula
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_non_terminal = 1.0 - last_done.float()
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            # TD residual: how much better was the actual reward + next value
            # compared to what the critic predicted?
            delta    = self.rewards[t] + CFG.gamma * next_value * next_non_terminal - self.values[t]

            # Accumulate GAE with exponential decay
            last_gae = delta + CFG.gamma * CFG.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.returns    = advantages + self.values
        self.advantages = advantages

    def mini_batches(self, minibatch_size):
        """
        Yield shuffled minibatches of flattened (T × E) transitions.

        WHY SHUFFLE: PPO's theoretical justification assumes i.i.d. samples.
        Sequential rollout data is highly correlated. Shuffling breaks that
        correlation so the gradient estimates are less biased.

        NOTE: Shuffling breaks temporal order, which means the GRU memory we
        stored at collection time is the correct "past" for each (obs, action)
        pair — we're not running the GRU forward again, just re-scoring.
        """
        T, E = self.T, self.E
        total = T * E

        # Flatten T × E → total
        obs       = self.obs.view(total, -1)
        actions   = self.actions.view(total)
        log_probs = self.log_probs.view(total)
        returns   = self.returns.view(total)
        advantages= self.advantages.view(total)
        memories  = self.memories.view(total, -1)

        # Normalise advantages across the whole batch
        # WHY: Keeps the scale of the actor loss consistent regardless of
        # how large or small the raw reward signal happens to be this update.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield (
                obs[idx],
                actions[idx],
                log_probs[idx],
                returns[idx],
                advantages[idx],
                memories[idx],
            )

    def reset(self):
        self.step = 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

CFG = Config()  # single global config instance referenced by RolloutBuffer too


def train():
    import os
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    print(f"Device: {CFG.device}")
    print(f"Envs: {CFG.num_envs}  |  Rollout steps: {CFG.rollout_steps}")
    print(f"Total env steps: {CFG.total_steps:,}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    world = World(
        num_envs         = CFG.num_envs,
        grid_size        = CFG.grid_size,
        food_zones       = CFG.food_zones,
        min_food         = CFG.min_food,
        sensor_radius    = CFG.sensor_radius,
        fov_degrees      = CFG.fov_degrees,
        front_fov_radius = CFG.front_fov_radius,
        side_fov_radius  = CFG.side_fov_radius,
        max_life         = CFG.max_life,
        food_reward      = CFG.food_reward,
        life_decay       = CFG.life_decay,
        device           = CFG.device,
    )

    bug = BugActorCritic(
        num_sensors = world.num_sensors,
        hidden_dim  = CFG.hidden_dim,
    ).to(CFG.device)

    optimiser = torch.optim.Adam(bug.parameters(), lr=CFG.lr, eps=1e-5)

    buffer = RolloutBuffer(
        rollout_steps = CFG.rollout_steps,
        num_envs      = CFG.num_envs,
        obs_dim       = world.obs_dim,
        hidden_dim    = CFG.hidden_dim,
        device        = CFG.device,
    )

    # ── Initial state ─────────────────────────────────────────────────────────
    obs    = world.reset()
    memory = bug.init_memory(CFG.num_envs, CFG.device)

    total_env_steps = 0
    update_count    = 0
    t_start         = time.time()

    # Running stats for logging
    episode_rewards  = torch.zeros(CFG.num_envs, device=CFG.device)
    completed_rewards = []   # rewards of finished episodes this interval

    updates_needed = CFG.total_steps // (CFG.rollout_steps * CFG.num_envs)

    print(f"\nStarting training — {updates_needed} updates planned\n")

    # # HARD TEST: manually walk bug onto a food tile and check eating fires
    # obs = world.reset()

    # # Find a food tile on env 0
    # food_tiles = (world.map[0] == World.FOOD).nonzero(as_tuple=False)
    # print(f"Food tiles on env 0: {food_tiles[:3]}")  # [y, x] format

    # # Place bug one step north of a food tile, facing south
    # # So action=0 (forward) should walk it onto food
    # food_y, food_x = food_tiles[0][0].item(), food_tiles[0][1].item()
    # world.positions[0] = torch.tensor([food_x, food_y - 1], device=CFG.device)
    # world.headings[0]  = torch.tensor(World.SOUTH, device=CFG.device)

    # print(f"Bug at: {world.positions[0].tolist()}, heading SOUTH")
    # print(f"Food at: [{food_x}, {food_y}]")
    # print(f"Tile in front of bug: {world.map[0, food_y, food_x].item()}")

    # actions = torch.zeros(world.num_envs, dtype=torch.long, device=CFG.device)
    # obs, rewards, dones = world.step(actions)
    # print(f"After stepping forward — reward[0]: {rewards[0].item()}, life[0]: {world.life_force[0].item()}")

    # obs = world.reset()
    # print(f"Bug position: {world.positions[0].tolist()}")
    # print(f"Bug heading: {world.headings[0].item()}")
    # food_tiles = (world.map[0] == World.FOOD).nonzero(as_tuple=False)
    # print(f"Food locations [y,x]: {food_tiles.tolist()}")
    # print(f"Food in obs: {(obs[0, :-1] == World.FOOD).any().item()}")
    # print(f"Obs sensor values: {obs[0, :-1].tolist()}")

    for update in range(1, updates_needed + 1):

        # ── PHASE 1: ROLLOUT COLLECTION ───────────────────────────────────────
        # Run the current policy in the world for rollout_steps steps.
        # No gradients here — we're just collecting data.
        bug.eval()
        buffer.reset()

        with torch.no_grad():
            for _ in range(CFG.rollout_steps):
                action, log_prob, value, new_memory = bug(obs, memory)

                next_obs, reward, done = world.step(action)

                buffer.store(obs, action, log_prob, value, reward, done, memory)

                # Accumulate episode reward tracking
                episode_rewards += reward

                # Handle episode endings mid-rollout
                if done.any():
                    done_ids = done.nonzero(as_tuple=False).squeeze(1)

                    # Record the rewards for finished episodes
                    for eid in done_ids:
                        completed_rewards.append(episode_rewards[eid].item())
                    episode_rewards[done_ids] = 0.0

                    # Zero out memory for finished envs — fresh episode = fresh state
                    new_memory[done_ids] = 0.0

                    # Catch the newly generated observation buffer. It contains the 
                    # fresh spawns for the dead bugs, and the normal states for the living ones.
                    next_obs = world.reset(env_ids=done_ids)

                obs    = next_obs
                memory = new_memory

            # Bootstrap: what does the critic think the current state is worth?
            # This is the V(s_T) term that handles rollout truncation.
            _, _, last_value, _ = bug(obs, memory)

        total_env_steps += CFG.rollout_steps * CFG.num_envs

        # ── PHASE 2: COMPUTE RETURNS ──────────────────────────────────────────
        buffer.compute_returns(last_value, done)

        # ── PHASE 3: PPO UPDATE ───────────────────────────────────────────────
        # Run ppo_epochs passes of minibatch gradient descent over the buffer.
        bug.train()

        actor_losses  = []
        value_losses  = []
        entropy_vals  = []
        clip_fractions = []

        for _ in range(CFG.ppo_epochs):
            for batch in buffer.mini_batches(CFG.minibatch_size):
                b_obs, b_actions, b_old_log_probs, b_returns, b_advantages, b_memories = batch

                # Re-score the stored actions under the current policy
                new_log_probs, entropy, new_values = bug.evaluate_actions(
                    b_obs, b_memories, b_actions
                )

                # ── Actor loss (clipped surrogate objective) ──────────────────
                # ratio > 1: new policy thinks this action is more likely than before
                # ratio < 1: new policy thinks it's less likely
                ratio      = (new_log_probs - b_old_log_probs).exp()
                clipped    = ratio.clamp(1.0 - CFG.clip_eps, 1.0 + CFG.clip_eps)

                # Take the more pessimistic of clipped/unclipped
                actor_loss = -torch.min(ratio * b_advantages, clipped * b_advantages).mean()

                # ── Value loss ────────────────────────────────────────────────
                # Train the critic to predict actual returns
                value_loss = F.mse_loss(new_values, b_returns)

                # ── Entropy bonus ─────────────────────────────────────────────
                # Negative because we want to MAXIMISE entropy (more exploration)
                entropy_loss = -entropy.mean()

                # ── Combined loss ─────────────────────────────────────────────
                loss = actor_loss + CFG.value_coeff * value_loss + CFG.entropy_coeff * entropy_loss

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bug.parameters(), CFG.max_grad_norm)
                optimiser.step()

                # Track diagnostics
                actor_losses.append(actor_loss.item())
                value_losses.append(value_loss.item())
                entropy_vals.append(-entropy_loss.item())
                clip_fractions.append(((ratio - 1.0).abs() > CFG.clip_eps).float().mean().item())

        update_count += 1

        # ── LOGGING ───────────────────────────────────────────────────────────
        if update % CFG.log_interval == 0:
            steps_per_sec = total_env_steps / (time.time() - t_start)
            mean_ep_reward = (
                sum(completed_rewards[-100:]) / max(len(completed_rewards[-100:]), 1)
            )

            print(
                f"Update {update:5d} | "
                f"Steps {total_env_steps:>10,} | "
                f"SPS {steps_per_sec:>7.0f} | "
                f"Ep reward (last 100): {mean_ep_reward:>7.2f} | "
                f"Actor loss: {sum(actor_losses)/len(actor_losses):>7.4f} | "
                f"Value loss: {sum(value_losses)/len(value_losses):>7.4f} | "
                f"Entropy: {sum(entropy_vals)/len(entropy_vals):>6.4f} | "
                f"Clip frac: {sum(clip_fractions)/len(clip_fractions):>.3f}"
            )
            completed_rewards.clear()

        # ── CHECKPOINT ────────────────────────────────────────────────────────
        if update % CFG.save_interval == 0:
            path = f"{CFG.checkpoint_dir}/bug_update_{update:06d}.pt"
            torch.save({
                "update":      update,
                "total_steps": total_env_steps,
                "model":       bug.state_dict(),
                "optimiser":   optimiser.state_dict(),
            }, path)
            print(f"  → Saved checkpoint: {path}")

    # ── Final save ────────────────────────────────────────────────────────────
    torch.save({
        "update":      update_count,
        "total_steps": total_env_steps,
        "model":       bug.state_dict(),
        "optimiser":   optimiser.state_dict(),
    }, f"{CFG.checkpoint_dir}/bug_final.pt")

    total_time = time.time() - t_start
    print(f"\nTraining complete — {total_env_steps:,} steps in {total_time:.1f}s")
    print(f"Average SPS: {total_env_steps / total_time:.0f}")


if __name__ == "__main__":
    train()