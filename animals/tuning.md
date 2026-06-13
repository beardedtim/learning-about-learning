# 🧠 BugBrain Matrix: PPO & Hyperparameter Tuning Guide

Reinforcement Learning (PPO) is highly sensitive to the balance of its hyperparameters. If the bug is acting stupid, spinning in circles, or collapsing after learning, one of these knobs is likely turned too high or too low.

Use this guide to diagnose behaviors and iterate on your `PPOConfig` and `BrainConfig`.

---

## 1. The Dopamine System (Exploration vs. Exploitation)

### `ent_coef` (Entropy Coefficient)

- **What it is:** A penalty applied to the network if it becomes too certain of its actions. It forces the bug to keep trying random moves (exploration).
- **Default / Safe Range:** `0.001` to `0.01`.
- **When to INCREASE it:** The bug finds one bad strategy (e.g., "hug the left wall") and refuses to try anything else. It stops learning early because it lacks the curiosity to find the food.
- **When to DECREASE it:** The policy collapses. The bug learns how to find food, but suddenly goes back to acting completely random. A high `ent_coef` violently punishes the bug for knowing exactly what to do.

### `int_coef` (Intrinsic Curiosity / RND Coefficient)

- **What it is:** The multiplier for your RND novelty reward. It dictates how much the bug cares about "seeing new things" vs. "eating actual food."
- **Default / Safe Range:** `0.005` to `0.05`.
- **When to INCREASE it:** The map is massive, the food is incredibly sparse, and the bug dies of the time-tax before ever stumbling into the food. It needs more internal motivation to walk around.
- **When to DECREASE it:** The bug ignores food entirely to go stare at corners it hasn't seen yet. If curiosity is paying out more than eating, this knob is too high.

---

## 2. Time & Horizons (How the Bug Perceives the Future)

### `rollout_steps`

- **What it is:** How many steps the bug takes in the environment before pausing to update its brain (calculating advantages and backpropagating).
- **Default / Safe Range:** `128`, `256`, or `512`.
- **When to INCREASE it:** Episodes (lifespans) are very long. If a bug lives for 1000 steps but rollout is 128, the LSTM memory gets awkwardly chopped up, and the bug struggles to link an action taken at step 50 to a reward received at step 200.
- **When to DECREASE it:** The bug takes forever to learn. Smaller rollouts mean more frequent brain updates.

### `gamma` ($\gamma$) (Discount Factor)

- **What it is:** How much the bug cares about the future. A value of `0.99` means a reward 100 steps from now is still highly motivating. A value of `0.50` means the bug only cares about the next few steps.
- **Default / Safe Range:** `0.99` (Standard) or `0.999` (Very long horizons).
- **When to INCREASE it:** The bug easily eats food right next to it, but won't travel across the map for a massive food pile.
- **When to DECREASE it:** The bug is paralyzed by overthinking the distant future and fails to secure the easy, immediate food right in front of it.

### `gae_lambda` ($\lambda$) (Advantage Smoothing)

- **What it is:** Balances how much we trust the Critic's predictions vs. the actual rewards received.
- **Default / Safe Range:** `0.95`.
- **When to tweak:** Rarely touch this. If your Critic is terribly inaccurate, lower it to `0.90` to rely more on actual observed rewards.

---

## 3. The Learning Engine (Gradient Descent Dynamics)

### `lr` (Learning Rate)

- **What it is:** How drastically the neural network weights change after every batch.
- **Default / Safe Range:** `1e-4` to `3e-4`.
- **When to INCREASE it:** The loss curves are completely flat. The bug has been training for 2 million steps and is still acting like it was just born.
- **When to DECREASE it:** Loss curves look like chaotic seismograph readings. The bug learns, forgets, learns, forgets. The updates are too violent.

### `ppo_epochs`

- **What it is:** How many times PPO loops over the same `rollout_steps` batch to squeeze learning out of it before throwing the data away.
- **Default / Safe Range:** `3` to `8`.
- **When to INCREASE it:** You want better sample efficiency (learning more from less environment interaction).
- **When to DECREASE it:** The bug overfits to recent experiences. It learns a trick, does it for one rollout, updates its brain 10 times, and completely forgets how to do anything else.

### `clip_coef` (PPO Clip Range)

- **What it is:** The magic of PPO. It prevents the policy from changing more than this percentage (e.g., `0.2` = 20%) in a single update, preventing catastrophic forgetting.
- **Default / Safe Range:** `0.1` to `0.2`.
- **When to INCREASE it:** You have a very stable, deterministic environment and want the bug to learn faster.
- **When to DECREASE it:** The bug's behavior is highly unstable. Dropping to `0.1` forces the network to take baby steps when changing its mind.

---

## 4. Brain Architecture (The LSTM Memory)

### `num_layers` (LSTM Depth)

- **What it is:** How many LSTMs are stacked on top of each other.
- **Default / Safe Range:** `1`.
- **Why keep it at 1:** RL gradients are incredibly fragile. Pushing gradients backwards through time AND through 3 vertical layers usually results in exploding gradients and total brain death. Only increase to `2` if the bug is completely incapable of navigating complex mazes.

### `hidden_dim`

- **What it is:** The number of neurons in the LSTM memory vector. It represents the "width" of the bug's working memory.
- **Default / Safe Range:** `128` to `256`.
- **When to INCREASE it:** The observation space is huge (e.g., raycasts, large grids, complex internal stats) and the bug needs more capacity to hold it all.
- **When to DECREASE it:** Training is too slow, or the bug is memorizing specific map seeds instead of learning general rules.

## 5. Reading the Matrix (Understanding Your Debug Logs)

When your console prints a line like this:
`Update  150/1220 | Food/Ep:  6.2 | EpLen: 358.1 | IntRwd:   1.31 | V-Ext: 225.99 | V-Int:   0.00 | RND:  0.003 | PG: 0.001 | Entropy: 0.90`

It is giving you a real-time x-ray of the bug's psychology. Here is how to translate those numbers into behavioral insights:

### The Physical World (Is the bug surviving?)

- **`Food/Ep` (Food per Episode) & `ep_reward`:** The ultimate metric of success. If this number is going up, your bug is learning.
- **`EpLen` (Episode Length):** How many steps the bug survives. Since `max_life_force` is ~200, an `EpLen` of 358 proves that eating food is successfully extending their lifespans! If `EpLen` ever drops to exactly your max life force (or lower), the bugs are starving to death.
- **`current_avg_life_force` vs `max_life_force`:** Shows the health of the population at the moment the log printed. If average life is hovering around 120/200, it means they are consistently finding food before dipping too close to death.

### The Dopamine System (What is the bug feeling?)

- **`IntRwd` (Intrinsic Reward):** The raw curiosity signal before your `int_coef` scales it down.
- **`V-Ext` (Value Loss Extrinsic):** How surprised the bug's brain was by the actual food it found.
- **`V-Int` (Value Loss Intrinsic):** How surprised the critic was by the curiosity reward.

### The Brain Metrics (How stable is the network?)

- **`Entropy` (Confidence/Randomness):** For an action space of 3 (Left, Right, Forward), the maximum possible entropy (complete total guessing) is roughly **1.10**. If this number crashes to `0.1`, the bug is hyper-fixated on a single action (e.g., holding the "Forward" button forever).
- **`PG` (Policy Gradient Loss):** How drastically the actor's weights are changing. You want this hovering very close to `0.000`. If this number spikes to `0.1` or `-0.1`, the network just experienced a catastrophic gradient update and likely destroyed its own memories.
- **`RND` (RND Loss):** How well the Predictor network is matching the Target network. As the bug explores the whole map, this number should slowly decay toward zero.
