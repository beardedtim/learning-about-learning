# Architecture Reference: PPO Training Environment (`train.py`)

This document provides a low-level technical analysis of the Proximal Policy Optimization (PPO) training loop implemented in `train.py`. The pipeline is engineered to optimize a recurrent actor-critic policy under memory-fused conditions, handling explicit hidden-state resets and a dual-reward structure (extrinsic survival and intrinsic curiosity).

---

## 1. Core Training Pipeline Overview

The training script executes a synchronized three-phase execution cycle over a fixed number of global updates:

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Trajectory Rollout (Data Collection)                          │
│ - Collect transitions step-by-step using current policy \pi_\theta     │
│ - Mask LSTM hidden states (h, c) on environment episode closures      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Dual Generalized Advantage Estimation (GAE)                   │
│ - Compute backwards-in-time temporal difference target streams         │
│ - Separate Extrinsic (Survival) and Intrinsic (Curiosity) baselines    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Mini-batch Policy & RND Optimization                          │
│ - Replay complete trajectories through vectorized sequence unrolling   │
│ - Compute clipped surrogate losses and update network weights          │
└────────────────────────────────────────────────────────────────────────┘

```

---

## 2. Low-Level Phase Analysis & Design Rationale

### Phase 1: Trajectory Rollout & Memory Protection

During data collection, the environment is stepped sequentially for a fixed horizon defined by `rollout_steps`.

- **Pre-allocated GPU Buffers:**
  All transition tensors (`b_obs`, `b_actions`, `b_logprobs`, `b_rewards`, `b_values`, `b_dones`) are allocated directly on the designated hardware accelerator (`world_cfg.device`) prior to loop entry. This eliminates redundant Host-to-Device (CPU-to-GPU) memory transfers during execution, forcing the environment step loop to run at maximum memory bandwidth.
- **Mid-Sequence Trajectory Wiping:**
  When parallel environments flag an episode termination (`dones=True`), the corresponding batch slice in the upcoming working hidden states must be instantaneously zeroed:

```python
new_h[:, dones.bool(), :] = 0.0
new_c[:, dones.bool(), :] = 0.0

```

If this memory clear were omitted, the LSTM would carry over historical context from the previous, deceased agent into the start of a completely fresh agent's life. This would corrupt the recurrent gradients, falsely teaching the policy that long-term past events dictate current state transitions across death boundaries.

---

### Phase 2: Dual Generalized Advantage Estimation (GAE)

Because the agent operates in a highly localized POMDP, environmental rewards are exceptionally sparse. To stabilize training, we employ Generalized Advantage Estimation (GAE), but split it into separate channels to handle the vastly different scales of extrinsic survival and intrinsic exploration.

The generalized advantage estimate $A_t$ for a given reward stream is calculated backwards from the horizon $T$:

$$A_t = \delta_t + (\gamma \lambda) \cdot \text{nextnonterminal} \cdot A_{t+1}$$

Where the single-step temporal difference error $\delta_t$ is defined as:

$$\delta_t = r_t + \gamma \cdot \text{nextvalues} \cdot \text{nextnonterminal} - V(s_t)$$

- **The Separate Horizon Rationale:**
- **Extrinsic GAE ($\gamma_{\text{ext}} = 0.99$):** Models immediate, physics-bound realities like eating or taking damage.
- **Intrinsic GAE ($\gamma_{\text{int}} = 0.99$):** Models structural novelty. Intrinsic values decay as a region becomes familiar, so keeping a distinct value baseline prevents the policy from over-indexing on temporary curiosity spikes at the cost of long-term metabolic survival.

- **Advantage Combination:**
  The final optimization advantage vector is a weighted composition of both streams:

```python
advantages = advantages_ext + (0.5 * advantages_int)

```

This forces the gradient steps to favor actions that simultaneously preserve life-force while driving the agent toward unmapped, high-novelty state regions.

---

### Phase 3: Neural Network Optimization

Once data collection finishes, the script transitions from step-by-step execution to vectorized sequence processing.

- **Sequence-First Tensor Reorganization:**
  The buffers are reshaped using a tensor transposition:

```python
env_obs = b_obs.transpose(0, 1)  # Re-orders from (T, B, F) to (B, T, F)

```

This flips the dimensions so that the batch size ($B$) represents individual environments, and the sequence length ($T$) represents consecutive time steps. Passing this block directly to `brain.evaluate_actions` triggers high-performance sequence tracking, allowing the network to process the entire rollout window in a single pass.

- **Clipped Surrogate Objective Optimization:**
  To prevent the updated policy $\pi_\theta$ from diverging too far from the historical rollout policy $\pi_{\theta_{\text{old}}}$, the policy gradient loss implements a strict probability ratio constraint $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$:

$$L^{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

- **Curiosity Predictor Optimization:**
  During each optimization epoch, the Random Network Distillation (RND) predictor network is updated directly using the stored states:

```python
rnd_loss = brain.compute_rnd_loss(env_obs)

```

As the predictor minimizes its error against the frozen target network for frequently visited states, the resulting `rnd_loss` drops. This systematically dampens the intrinsic reward for familiar visual configurations, naturally forcing the agent's focus toward unvisited territory.

---

## 3. Hyperparameter Rationale

| Hyperparameter           | Value           | Structural Purpose                                                                                                                                              |
| ------------------------ | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rollout_steps`          | `128` / `256`   | Sets the recurrent temporal window. Longer sequences allow the LSTM to capture deeper historical relationships, but demand higher GPU VRAM.                     |
| `clip_coef` ($\epsilon$) | `0.2`           | Bounds the maximum policy update step size, preventing catastrophic policy degradation during noisy curiosity phases.                                           |
| `ent_coef`               | `0.01` / `0.02` | Scales the Shannon Entropy bonus. Forces uniform exploration early in training, ensuring the agent tries diverse action patterns before converging on a policy. |
| `vf_coef`                | `0.5`           | Weights the value network losses relative to the policy loss, ensuring stable critic convergence without dominating the overall policy gradient.                |
| `eps` (`1e-5`)           | `1e-5`          | Added to the Adam optimizer denominator to ensure numerical stability and prevent division-by-zero errors during high-frequency weight updates.                 |
