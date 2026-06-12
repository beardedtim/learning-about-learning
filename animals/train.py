import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from dataclasses import dataclass

from world import BiomeConfig, World, WorldConfig, get_sensors
from brains import ActorCriticBrain, build_obs_features, NUM_CELL_TYPES, ACTION_DIM
from log import setup_logger

# ==========================================
# CONST: PPO GLOBAL DEFAULTS
# ==========================================
DEFAULT_ROLLOUT_STEPS = 128
DEFAULT_PPO_EPOCHS = 4
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_COEF = 0.2
DEFAULT_ENT_COEF = 0.01
DEFAULT_VF_COEF = 0.5
DEFAULT_LR = 3e-4

@dataclass
class PPOConfig:
    """
    Configuration for the Proximal Policy Optimization training loop.
    """
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS
    ppo_epochs: int = DEFAULT_PPO_EPOCHS
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_coef: float = DEFAULT_CLIP_COEF
    ent_coef: float = DEFAULT_ENT_COEF
    vf_coef: float = DEFAULT_VF_COEF
    lr: float = DEFAULT_LR

def train_ppo(world_cfg: WorldConfig, ppo_cfg: PPOConfig = PPOConfig(), save_path="smart_bug.pt", load_path="smart_bug.pt", total_timesteps=1_000_000, layout="easy"):
    logger = setup_logger()
    # --- PPO Hyperparameters ---
    rollout_steps = ppo_cfg.rollout_steps
    ppo_epochs = ppo_cfg.ppo_epochs
    gamma = ppo_cfg.gamma
    gae_lambda = ppo_cfg.gae_lambda
    clip_coef = ppo_cfg.clip_coef
    ent_coef = ppo_cfg.ent_coef
    vf_coef = ppo_cfg.vf_coef
    lr = ppo_cfg.lr

    num_updates = total_timesteps // (world_cfg.envs * rollout_steps)

    # 1. Initialize Environment & Brain
    env = World(world_cfg)

    raw_obs = env.reset(layout=layout).squeeze(1)
    prev_action = torch.zeros(world_cfg.envs, dtype=torch.long, device=world_cfg.device)
    prev_reward = torch.zeros(world_cfg.envs, device=world_cfg.device)
    obs = build_obs_features(raw_obs, prev_action, prev_reward, env.life_force.squeeze(1), max_life_force=world_cfg.max_life_force)
    V = env.obs_size
    obs_dim = NUM_CELL_TYPES * V + ACTION_DIM + 2

    brain = ActorCriticBrain(obs_dim=obs_dim, action_dim=3).to(world_cfg.device)

    # --- Checkpoint Loading ---
    if os.path.exists(load_path):
        logger.info(f"Found existing brain at '{load_path}'. Loading weights to resume training...")
        # map_location ensures it loads correctly even if you switch between CPU and CUDA
        brain.load_state_dict(torch.load(load_path, map_location=world_cfg.device))
    else:
        logger.info(f"No existing brain found at '{load_path}'. Initializing a fresh brain.")
    # -------------------------------

    optimizer = optim.Adam(brain.parameters(), lr=lr, eps=1e-5)
    # Initial hidden state
    h, c = brain.init_hidden(batch_size=world_cfg.envs, device=world_cfg.device)

    # 2. Pre-allocate Rollout Buffers on the GPU (Zero memory transfers!)
    b_obs = torch.zeros((rollout_steps, world_cfg.envs, obs_dim), device=world_cfg.device)
    b_actions = torch.zeros((rollout_steps, world_cfg.envs), dtype=torch.long, device=world_cfg.device)
    b_logprobs = torch.zeros((rollout_steps, world_cfg.envs), device=world_cfg.device)
    b_rewards = torch.zeros((rollout_steps, world_cfg.envs), device=world_cfg.device)
    b_values = torch.zeros((rollout_steps, world_cfg.envs), device=world_cfg.device)
    b_dones = torch.zeros((rollout_steps, world_cfg.envs), device=world_cfg.device)
    b_int_rewards = torch.zeros((rollout_steps, world_cfg.envs), device=world_cfg.device)
    b_values_int = torch.zeros((rollout_steps, world_cfg.envs), device=world_cfg.device)
    dones = torch.zeros(world_cfg.envs, device=world_cfg.device)
    logger.info(f"Starting Training: {num_updates} updates of {world_cfg.envs * ppo_cfg.rollout_steps} steps each.")


    ep_returns = torch.zeros(world_cfg.envs, device=world_cfg.device)
    ep_lengths = torch.zeros(world_cfg.envs, device=world_cfg.device)
    completed_returns = []
    completed_lengths = []
    for update in range(1, num_updates + 1):
        # Save the hidden state at the START of the rollout to replay during training
        initial_h, initial_c = h.clone(), c.clone()

        # ==========================================
        # PHASE 1: DATA COLLECTION (ROLLOUT)
        # ==========================================
        for step in range(rollout_steps):
            b_obs[step] = obs
            b_dones[step] = dones

            # Get action from brain
            with torch.no_grad():
                action_logits, value_ext, value_int, (new_h, new_c) = brain(obs, (h, c))
                dist = Categorical(logits=action_logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            # Store actions and values (using the extrinsic critic for now)
            b_actions[step] = actions
            b_logprobs[step] = logprobs
            b_values[step] = value_ext.squeeze(-1)
            b_values_int[step] = value_int.squeeze(-1)

            # Step the environment
            next_obs, rewards, next_dones = env.step(actions.unsqueeze(1))

            # Format outputs
            raw_obs = next_obs.squeeze(1)
            rewards = rewards.squeeze(1)
            rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
            dones = next_dones.squeeze(-1) if next_dones.dim() > 1 else next_dones
            b_rewards[step] = rewards
            
            ep_returns += rewards
            ep_lengths += 1

            for i in range(world_cfg.envs):
                if dones[i]:
                    completed_returns.append(ep_returns[i].item())
                    completed_lengths.append(ep_lengths[i].item())
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0.0

            prev_action = actions
            prev_reward = rewards

            # 1. Wipe memory for dead bugs FIRST
            if dones.any():
                new_h[:, dones.bool(), :] = 0.0
                new_c[:, dones.bool(), :] = 0.0
                prev_action[dones] = 0
                prev_reward[dones] = 0.0

            # 2. NOW build the features for the next step
            obs = build_obs_features(raw_obs, prev_action, prev_reward, env.life_force.squeeze(1), max_life_force=world_cfg.max_life_force)

            h, c = new_h, new_c

            # CALCULATE INTRINSIC REWARD ON THE NEW STATE
            with torch.no_grad():
                target_feat = brain.rnd_target(obs)
                pred_feat = brain.rnd_predictor(obs)
                # The MSE between predictor and target is the novelty score
                int_reward = torch.mean((pred_feat - target_feat)**2, dim=-1)
            
            b_int_rewards[step] = int_reward

            # Wipe memory for dead bugs
            if dones.any():
                new_h[:, dones.bool(), :] = 0.0
                new_c[:, dones.bool(), :] = 0.0
                prev_action[dones] = 0
                prev_reward[dones] = 0.0

            h, c = new_h, new_c

        # ==========================================
        # PHASE 2: CALCULATE ADVANTAGES (GAE)
        # ==========================================
        with torch.no_grad():
            _, next_value_ext, next_value_int, _ = brain(obs, (h, c))
            next_value = next_value_ext.squeeze(-1)
            next_value_int = next_value_int.squeeze(-1) # GET LAST INT VALUE

            advantages_ext = torch.zeros_like(b_rewards, device=world_cfg.device)
            advantages_int = torch.zeros_like(b_rewards, device=world_cfg.device)
            lastgaelam_ext = 0
            lastgaelam_int = 0

            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    nextnonterminal = 1.0 - dones.float()
                    nextvalues_ext = next_value
                    nextvalues_int = next_value_int
                else:
                    nextnonterminal = 1.0 - b_dones[t + 1].float()
                    nextvalues_ext = b_values[t + 1]
                    nextvalues_int = b_values_int[t + 1]

                # Extrinsic GAE
                delta_ext = b_rewards[t] + gamma * nextvalues_ext * nextnonterminal - b_values[t]
                advantages_ext[t] = lastgaelam_ext = delta_ext + gamma * gae_lambda * nextnonterminal * lastgaelam_ext

                # Intrinsic GAE (often uses a higher gamma like 0.99 for longer curiosity horizons)
                delta_int = b_int_rewards[t] + 0.99 * nextvalues_int * nextnonterminal - b_values_int[t]
                advantages_int[t] = lastgaelam_int = delta_int + 0.99 * gae_lambda * nextnonterminal * lastgaelam_int

            advantages = advantages_ext + (0.01 * advantages_int)
            returns_ext = advantages_ext + b_values
            returns_int = advantages_int + b_values_int

        # ==========================================
        # PHASE 3: OPTIMIZE NEURAL NETWORK
        # ==========================================
        env_obs = b_obs.transpose(0, 1)       
        env_actions = b_actions.transpose(0, 1) 
        env_dones = b_dones.transpose(0, 1)

        flat_advantages = advantages.transpose(0, 1).flatten()
        flat_returns_ext = returns_ext.transpose(0, 1).flatten()
        flat_returns_int = returns_int.transpose(0, 1).flatten()
        flat_old_logprobs = b_logprobs.transpose(0, 1).flatten()

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        for epoch in range(ppo_epochs):
            # 1. Pass the whole sequence through the brain at once
            new_logprobs, new_values_ext, new_values_int, entropy = brain.evaluate_actions(
                env_obs,
                (initial_h, initial_c),
                env_actions,
                env_dones
            )

            # Flatten outputs to match advantages/returns
            new_logprobs = new_logprobs.flatten()
            new_values_ext = new_values_ext.flatten()
            new_values_int = new_values_int.flatten()

            # 2. Calculate PPO Ratio
            logratio = new_logprobs - flat_old_logprobs
            ratio = logratio.exp()

            # 3. Calculate Policy Loss
            pg_loss1 = -flat_advantages * ratio
            pg_loss2 = -flat_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # 4. Calculate Dual Value Loss
            v_loss_ext = 0.5 * ((new_values_ext - flat_returns_ext) ** 2).mean()
            v_loss_int = 0.5 * ((new_values_int - flat_returns_int) ** 2).mean()
            v_loss = v_loss_ext + v_loss_int

            # 5. Train the Curiosity Predictor! (THIS WAS LIKELY MISSING)
            rnd_loss = brain.compute_rnd_loss(env_obs)

            # 6. Total Loss (Make sure rnd_loss is added here!)
            loss = pg_loss - ent_coef * entropy + v_loss * vf_coef + rnd_loss

            # 7. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(brain.parameters(), 0.5) 
            optimizer.step()

        # --- Telemetry ---
        if update % 10 == 0:
            # 1. Calculate the true episodic averages from the dead bugs
            safe_mean = lambda x: sum(x[-100:]) / len(x[-100:]) if x else 0.0
            
            true_ep_reward = safe_mean(completed_returns)
            true_ep_length = safe_mean(completed_lengths)
            
            # 2. Calculate current snapshot stats
            avg_int_reward = b_int_rewards.sum(dim=0).mean().item() 
            current_life = env.life_force.mean().item()

            # 3. The console message you see in the terminal
            msg = (f"Update {update:4d}/{num_updates} | "
                   f"EpRwd: {true_ep_reward:5.1f} | EpLen: {true_ep_length:5.1f} | "
                   f"IntRwd: {avg_int_reward:6.2f} | "
                   f"V-Ext: {v_loss_ext.item():6.2f} | V-Int: {v_loss_int.item():6.2f} | "
                   f"RND: {rnd_loss.item():6.3f} | PG: {pg_loss.item():.3f}")

            # 4. The structured dictionary that gets saved to the JSON log file
            metrics = {
                "update": update,
                "step": update * world_cfg.envs * ppo_cfg.rollout_steps,
                
                # THE TRUE EPISODE STATS (Replaces the old avg_reward and avg_life)
                "ep_reward": round(true_ep_reward, 2),
                "ep_length": round(true_ep_length, 2),
                
                # INTRINSIC STATS
                "avg_int_reward": round(avg_int_reward, 4),
                
                # NETWORK LOSSES
                "value_loss_ext": round(v_loss_ext.item(), 4),
                "value_loss_int": round(v_loss_int.item(), 4),
                "rnd_loss": round(rnd_loss.item(), 4),
                "policy_loss": round(pg_loss.item(), 4),
                "entropy": round(entropy.item(), 4),
                
                # INSTANTANEOUS ENVIRONMENT STATE
                "current_avg_life_force": round(current_life, 2),
                "max_life_force": round(env.life_force.max().item(), 2)
            }

            # Pass the dictionary to the logger
            logger.info(msg, extra={"metrics": metrics})

            # Save a checkpoint every 10 updates
            torch.save(brain.state_dict(), save_path)

    torch.save(brain.state_dict(), save_path)
    logger.info(f"Training complete. Brain saved to {save_path}")


def crawl():
    # === CRAWL: prove "find food, don't circle" ===
    crawl_biome = BiomeConfig(
        x=2, y=2, width=11, height=11,    # covers most of a 15x15 grid -> food is everywhere
        food_refresh_rate=0.3,             # high refresh -> food rarlat_advantages = advantages.transpose(0, 1).flatten()
        eating_bonus=20.0,
    )

    world_cfg_crawl = WorldConfig(
        grid_size=32,
        envs=32,
        biomes=[crawl_biome],
        bug_sensors=get_sensors(),
        num_bugs=1,
        min_food=15,
        device='cuda',
    )

    ppo_cfg_crawl = PPOConfig(
        rollout_steps=256,
        ent_coef=0.02,
        ppo_epochs=4,
        lr=3e-4,
    )

    print(f"=== Booting BugBrain Matrix ===")
    print(f"Device: {world_cfg_crawl.device.upper()}")
    print(f"Parallel Worlds: {world_cfg_crawl.envs}")

    train_ppo(
        world_cfg=world_cfg_crawl,
        ppo_cfg=ppo_cfg_crawl,
        save_path="stage1_crawl.pt",
        load_path="stage1_crawl.pt",
        total_timesteps=20_000_000,
        layout="easy"
    )


def walk():
    # === WALK: same biome, but now behind a maze ===
    walk_biome = BiomeConfig(
        x=2, y=2, width=11, height=11,    # same reward structure as crawl on purpose
        food_refresh_rate=0.3,
        eating_bonus=20.0,
    )

    world_cfg_walk = WorldConfig(
        grid_size=20,
        envs=32,
        biomes=[walk_biome],
        bug_sensors=get_sensors(),
        num_bugs=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    ppo_cfg_walk = PPOConfig(
        rollout_steps=256,
        ent_coef=0.02,
        ppo_epochs=4,
        lr=3e-4,
    )

    print(f"=== Booting BugBrain Matrix ===")
    print(f"Device: {world_cfg_walk.device.upper()}")
    print(f"Parallel Worlds: {world_cfg_walk.envs}")

    train_ppo(
        world_cfg=world_cfg_walk,
        ppo_cfg=ppo_cfg_walk,
        save_path="stage2_walk_medium.pt",
        load_path="stage2_walk_medium.pt",
        # load_path="stage1_crawl.pt",
        total_timesteps=20_000_000,
        layout="medium"
    )

    # train_ppo(
    #     world_cfg=world_cfg_walk,
    #     ppo_cfg=ppo_cfg_walk,
    #     save_path="stage2_walk_hard.pt",   # new save_path, loads stage2_walk_medium.pt weights manually if needed
    #     load_path="staeg2_wakk_medium.pt",
    #     total_timesteps=300_000,
    #     layout="hard"
    # )

def run():
    # === RUN: multiple biomes with distinct reward profiles, hard maze ===

    # "Jackpot" zone: rare food, big payoff
    jackpot_biome = BiomeConfig(
        x=2, y=2, width=6, height=6,
        food_refresh_rate=0.02,   # food spawns rarely here
        eating_bonus=60.0,        # but it's worth a lot
    )

    # "Steady" zone: common food, small payoff
    steady_biome = BiomeConfig(
        x=16, y=2, width=6, height=6,
        food_refresh_rate=0.4,    # food spawns often
        eating_bonus=8.0,         # but each piece is worth little
    )

    # "Desert" zone: food almost never spawns, tiny payoff -- a clear "bad" zone
    desert_biome = BiomeConfig(
        x=9, y=16, width=6, height=6,
        food_refresh_rate=0.01,
        eating_bonus=2.0,
    )

    world_cfg_run = WorldConfig(
        grid_size=24,
        envs=32,
        biomes=[jackpot_biome, steady_biome, desert_biome],
        bug_sensors=get_sensors(),
        min_food=10,
        num_bugs=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    ppo_cfg_run = PPOConfig(
        rollout_steps=128,
        ppo_epochs=4,
        lr=3e-4,
    )

    print(f"=== Booting BugBrain Matrix ===")
    print(f"Device: {world_cfg_run.device.upper()}")
    print(f"Parallel Worlds: {world_cfg_run.envs}")

    train_ppo(
        world_cfg=world_cfg_run,
        ppo_cfg=ppo_cfg_run,
        save_path="stage3_run.pt",   # resume from stage2 weights by copying the file first
        load_path="stage2_walk_hard.pt",
        total_timesteps=1_000_000,
        layout="hard"
    )

if __name__ == '__main__':
    crawl()
    # walk()