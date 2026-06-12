"""
train.py — PPO training loop for VecBiomeEnv + ActorCriticBrain.

Everything stays on GPU. The champion (highest mean episode return so far)
is saved to checkpoints/champion.pt whenever a new best is found.
A separate process (viz.py) can read that file to render it live.

Run:
    python train.py
"""

import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from brains   import ActorCriticBrain
from world import VecBiomeEnv, OBS_DIM, ACTION_DIM, MAX_STEPS

# ── config ────────────────────────────────────────────────────────────────────
N_ENVS         = 256          # bugs training in parallel
N_STEPS        = 128          # rollout length before each update
EPOCHS         = 4
MINIBATCHES    = 8            # N_ENVS split into this many minibatches
CLIP_EPS       = 0.2
ENT_COEF       = 0.01
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5
LR             = 3e-4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
TOTAL_UPDATES  = 10_000
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"

# ── setup ─────────────────────────────────────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env   = VecBiomeEnv(N_ENVS, device=DEVICE, seed=42)
brain = ActorCriticBrain(obs_dim=OBS_DIM, action_dim=ACTION_DIM).to(DEVICE)
opt   = optim.Adam(brain.parameters(), lr=LR, eps=1e-5)

# hidden state lives across rollout steps — shape (num_layers, N, hidden_dim)
def fresh_hidden():
    z = lambda: torch.zeros(brain.num_layers, N_ENVS, brain.hidden_dim, device=DEVICE)
    return z(), z()

h, c = fresh_hidden()

# initial obs
obs = env._get_obs()

champion_return = -float("inf")
total_steps     = 0
t0              = time.time()

print(f"Training on {DEVICE}  |  {N_ENVS} envs × {N_STEPS} steps = "
      f"{N_ENVS*N_STEPS} transitions/update")

# ── training loop ─────────────────────────────────────────────────────────────
for update in range(1, TOTAL_UPDATES + 1):

    # ── rollout collection ────────────────────────────────────────────────────
    buf_obs      = torch.zeros(N_STEPS, N_ENVS, OBS_DIM,    device=DEVICE)
    buf_actions  = torch.zeros(N_STEPS, N_ENVS,              device=DEVICE, dtype=torch.long)
    buf_logprobs = torch.zeros(N_STEPS, N_ENVS,              device=DEVICE)
    buf_values   = torch.zeros(N_STEPS, N_ENVS,              device=DEVICE)
    buf_rewards  = torch.zeros(N_STEPS, N_ENVS,              device=DEVICE)
    buf_dones    = torch.zeros(N_STEPS, N_ENVS,              device=DEVICE)

    # detach hidden so we don't backprop through the rollout collection
    h_roll, c_roll = h.detach(), c.detach()

    for t in range(N_STEPS):
        with torch.no_grad():
            logits, values, (h_roll, c_roll) = brain(obs, (h_roll, c_roll))

        dist    = Categorical(logits=logits)
        actions = dist.sample()
        logprob = dist.log_prob(actions)

        buf_obs     [t] = obs
        buf_actions [t] = actions
        buf_logprobs[t] = logprob
        buf_values  [t] = values.squeeze(-1)

        obs, reward, done = env.step(actions)

        buf_rewards[t] = reward
        buf_dones  [t] = done.float()

        # reset hidden for finished envs
        finished = done.nonzero(as_tuple=True)[0]
        if finished.numel() > 0:
            h_roll[:, finished, :] = 0.0
            c_roll[:, finished, :] = 0.0

    total_steps += N_ENVS * N_STEPS

    # ── GAE ───────────────────────────────────────────────────────────────────
    with torch.no_grad():
        _, last_values, _ = brain(obs, (h_roll, c_roll))
        last_values = last_values.squeeze(-1)   # (N,)

    advantages = torch.zeros_like(buf_rewards)
    last_gae   = torch.zeros(N_ENVS, device=DEVICE)

    for t in reversed(range(N_STEPS)):
        nv   = last_values if t == N_STEPS-1 else buf_values[t+1]
        nd   = buf_dones[t]
        delta     = buf_rewards[t] + GAMMA * nv * (1-nd) - buf_values[t]
        last_gae  = delta + GAMMA * GAE_LAMBDA * (1-nd) * last_gae
        advantages[t] = last_gae

    returns = advantages + buf_values

    # flatten (N_STEPS, N_ENVS, ...) → (N_STEPS*N_ENVS, ...)
    b_obs  = buf_obs     .view(-1, OBS_DIM)
    b_act  = buf_actions .view(-1)
    b_lp   = buf_logprobs.view(-1)
    b_ret  = returns     .view(-1)
    b_adv  = advantages  .view(-1)

    b_adv  = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
    b_ret = (b_ret - b_ret.mean()) / (b_ret.std() + 1e-8)
    # ── PPO update ────────────────────────────────────────────────────────────
    batch_size  = N_ENVS * N_STEPS
    mb_size     = batch_size // MINIBATCHES

    pg_l, vf_l, ent_l = 0., 0., 0.

    for _ in range(EPOCHS):
        perm = torch.randperm(batch_size, device=DEVICE)
        for start in range(0, batch_size, mb_size):
            idx = perm[start:start+mb_size]
            mb_obs = b_obs[idx]
            mb_act = b_act[idx]
            mb_lp  = b_lp [idx]
            mb_ret = b_ret[idx]
            mb_adv = b_adv[idx]

            # treat each sample as independent (truncated BPTT at minibatch boundary)
            h0 = torch.zeros(brain.num_layers, mb_size, brain.hidden_dim, device=DEVICE)
            c0 = torch.zeros(brain.num_layers, mb_size, brain.hidden_dim, device=DEVICE)

            # evaluate_actions expects (batch, seq, obs) — use seq=1
            logits, values, _ = brain(mb_obs, (h0, c0))
            dist     = Categorical(logits=logits)
            new_lp   = dist.log_prob(mb_act)
            entropy  = dist.entropy().mean()

            ratio    = (new_lp - mb_lp).exp()
            pg1      = -mb_adv * ratio
            pg2      = -mb_adv * ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS)
            pg_loss  = torch.max(pg1, pg2).mean()
            vf_loss  = nn.functional.mse_loss(values.squeeze(-1), mb_ret)
            loss     = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(brain.parameters(), MAX_GRAD_NORM)
            opt.step()

            pg_l  += pg_loss.item()
            vf_l  += vf_loss.item()
            ent_l += entropy.item()

    n_updates = EPOCHS * MINIBATCHES
    pg_l /= n_updates; vf_l /= n_updates; ent_l /= n_updates

    # ── logging ───────────────────────────────────────────────────────────────
    mean_ret  = buf_rewards.sum(0).mean().item()
    sps       = total_steps / (time.time() - t0)

    print(
        f"upd {update:5d} | steps {total_steps:9d} | "
        f"ret {mean_ret:+6.3f} | pg {pg_l:+.4f} | "
        f"vf {vf_l:.4f} | ent {ent_l:.3f} | {sps:,.0f} sps"
    )

    # ── champion ──────────────────────────────────────────────────────────────
    if mean_ret > champion_return:
        champion_return = mean_ret
        path = os.path.join(CHECKPOINT_DIR, "champion.pt")
        torch.save({
            "update":          update,
            "total_steps":     total_steps,
            "mean_return":     mean_ret,
            "brain":           brain.state_dict(),
        }, path)
        # also write a small json sidecar so viz.py can read stats without
        # loading the full checkpoint
        with open(os.path.join(CHECKPOINT_DIR, "champion.json"), "w") as f:
            json.dump({"update": update, "total_steps": total_steps,
                       "mean_return": mean_ret}, f)
        print(f"  ★  new champion  ret={mean_ret:+.4f}  saved to {path}")