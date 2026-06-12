# Architecture Reference: ActorCriticBrain Core

This document provides a low-level technical decomposition of the `ActorCriticBrain` architecture implemented in `brains.py`. The design is structured specifically to solve a **Partially Observable Markov Decision Process (POMDP)** under a rigid operational constraint: **the agent has zero external or privileged global information.** It must survive, navigate, and map its environment entirely using localized, ego-centric sensory data and its own internal memory.

---

## 1. Architectural Blueprint

The system coordinates five distinct neural sub-modules to map local observations into actions, track future expected survival returns, model intrinsic novelty, and maintain a persistent spatio-temporal working memory.

### Static Exploration Module (Random Network Distillation)

Separated from the recurrent policy network to ensure that environmental familiarity is calculated as a pure, non-recurrent spatial feature:

```
  observation[t] ──► [RND Target (Fixed)]       ──► Feature Vector A
  observation[t] ──► [RND Predictor (Learned)]  ──► Feature Vector B ──► MSE Loss -> Intrinsic Reward

```

---

## 2. Low-Level Component Breakdown & Design Rationale

### A. Observation Feature Synthesis (`build_obs_features`)

The raw state representation from the environment is highly categorical and heterogeneous. Before entering the parameterized network, it must undergo non-destructive structural normalization.

- **Categorical De-biasing via One-Hot Encoding:**
  The spatial observation array contains integers representing discrete entities (`BLOCKED=-2`, `WALL=-1`, `EMPTY=0`, `FOOD=1`, `ANIMAL=2`). Passing raw integer values directly into a linear layer implies a continuous metric space where $\text{WALL} < \text{EMPTY} < \text{FOOD}$, forcing the network to waste capacity trying to unlearn non-existent linear correlations. Shifting by `-min_cell_code` and casting to a 5-dimensional one-hot vector treats each environmental state as an independent orthogonal basis vector.
- **Recurrent Sensorimotor Loops:**
  The concatenation of the agent's `prev_action` (one-hot) and `prev_reward` (scalar) directly into the feature vector provides the memory core with a localized historical context. This allows the network to learn differential relationships (e.g., _"Moving forward yielded a reward drop; turning left is highly correlated with an increase in local food density"_).
- **Bounded Internal Telemetry:**
  The agent's internal `life_force` is scaled against `max_life_force` to map it directly into $[0, 1]$. This prevents activation explosions and serves as a vital internal metabolic signal, allowing the policy to shift dynamically from exploratory behavior (high life force) to high-risk foraging behavior (low life force).

### B. The Sensory Encoder

```python
self.encoder = nn.Sequential(
    nn.Linear(obs_dim, embed_dim),
    nn.LayerNorm(embed_dim),
    nn.ELU(),
    nn.Linear(embed_dim, embed_dim),
    nn.LayerNorm(embed_dim),
    nn.ELU(),
)

```

- **Why LayerNorm instead of BatchNorm?**
  Batch Normalization calculates statistics across the batch dimension. In recurrent reinforcement learning, data dependencies within trajectories break the Independent and Identically Distributed (I.I.D.) assumption. Furthermore, during sequence evaluation, batch statistics can fluctuate wildly depending on sequence lengths. Layer Normalization operates strictly across the channel/feature dimension per token, ensuring invariant activation scaling regardless of batch configuration or time-step index.
- **Why ELU instead of ReLU?**
  The Exponential Linear Unit (ELU) retains a non-zero gradient for negative inputs ($\alpha(e^x - 1)$). This minimizes the risk of "dead neurons"—a common failure mode in PPO where large policy updates can push weights into a region where ReLU outputs exactly zero across the entire dataset, permanently killing gradient flow through that sub-pathway.

### C. The Recurrent Memory Core (`nn.LSTM`)

The agent maps local observations using an `nn.LSTM` operating with `batch_first=True`. LSTMs are mathematically superior to GRUs for long-horizon sparse-reward exploration due to the decoupling of the hidden and cell states.

The hidden state $h_t$ represents immediate, high-frequency working memory used directly by the policy head to emit actions. The cell state $c_t$ serves as a dedicated internal gradient highway governed by explicit gating mechanisms:

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & \text{(Forget Gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & \text{(Input Gate)} \\
\tilde{c}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) & \text{(Candidate Cell State)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t & \text{(Cell State Update)}
\end{aligned}
$$

Because the update to the cell state $c_t$ is fundamentally additive via the forget gate $f_t$, the error gradient can propagate backwards through hundreds of steps without undergoing the exponential decay (vanishing gradient) typical of standard recurrent formulations. This allows the bug to connect distant past events (e.g., entering a dead-end corridor 80 steps ago) with current outcomes.

### D. The Dual-Critic Paradigm

```python
self.critic_ext = nn.Sequential(...) # Extrinsic Value Function
self.critic_int = nn.Sequential(...) # Intrinsic Value Function

```

Tracking environment rewards (food consumption, survival) and exploration rewards (novelty) requires two separate critic networks.

Extrinsic rewards are bounded, dense, and highly directional, reflecting the environmental physics. Intrinsic rewards are non-stationary, transient, and steadily decay as the environment becomes familiar. If these two optimization criteria were compressed into a single value network $V(s)$, the massive, volatile scales of early intrinsic exploration bonuses would warp the value baseline, completely masking the subtle, high-frequency extrinsic reward signals needed to learn basic survival mechanics. Separating them into $V_{\text{ext}}(s)$ and $V_{\text{int}}(s)$ isolates the gradient updates and stabilizes baseline estimation.

### E. Random Network Distillation (RND) Core

```python
self.rnd_target = nn.Sequential(...)    # Deterministic, Frozen Weights
self.rnd_predictor = nn.Sequential(...) # Trainable Weights

```

To prevent the agent from getting trapped in cyclical local behaviors or freezing in a corner out of fear, it must be rewarded for encountering unfamiliar spatial states.

An unparameterized count-based exploration metric cannot scale to large or continuous observation spaces. RND solves this by defining a fixed, randomly initialized neural network (`rnd_target`) that maps an observation to a lower-dimensional continuous embedding. A secondary network (`rnd_predictor`) is trained via gradient descent to match the exact output of the target network on states the agent visits.

- **Novelty Estimation:** If the agent encounters a highly familiar state, the predictor network will have seen similar inputs frequently and will easily mimic the target network, resulting in a low prediction error. If the agent enters an unvisited room or sees an uncommon pattern of obstacles, the predictor's output will diverge significantly from the target, generating a high Mean Squared Error (MSE). This error is isolated and passed directly to the agent as an intrinsic reward.
- **The Non-Recurrent Constraint:** Notice that both the `rnd_target` and `rnd_predictor` are strictly feed-forward networks that read raw observations, completely bypassed by the LSTM memory core. **This is an explicit design requirement.** If the RND networks were recurrent, the predictor could minimize its error simply by predicting features based on temporal context rather than spatial novelty. A state would stop being "novel" merely because the bug had looked at it for 5 consecutive frames. Non-recurrent RND forces the agent to map true spatial unfamiliarity independent of time.

---

## 3. Mathematical Flow of Evaluation (`evaluate_actions`)

During PPO optimization epochs, the sequential loop over time-steps must explicitly manage trajectory boundaries while maximizing memory layout efficiencies.

### Trajectory-Masked Hidden State Propagation

When the agent transitions to a `done` state mid-sequence, the internal hidden configurations must be immediately cleared to prevent information from bleeding into the next independent episode. This is executed via vectorized tensor multiplication before stepping the LSTM cell:

$$\text{mask}_t = 1.0 - \text{dones}[:, t]$$

$$h_t = h_{t-1} \odot \text{mask}_t$$

$$c_t = c_{t-1} \odot \text{mask}_t$$

### Probability Distribution and Entropy Regularization

Once the sequence has been processed through the recurrent core, the output states are projected to action logits. These logits parameterize a categorical distribution over the discrete action space:

$$P(a_i) = \frac{e^{\text{logit}_i}}{\sum_{j} e^{\text{logit}_j}}$$

The policy's log-probabilities are evaluated against the historical action indices collected during the rollout phase to calculate the PPO surrogate loss ratio. Simultaneously, the framework computes the policy's Shannon Entropy $\mathcal{H}$:

$$\mathcal{H}(\pi(s)) = -\sum_{i=1}^{A} P(a_i) \log P(a_i)$$

Maximizing the mean entropy $\mathcal{H}$ acts as an information-theoretic regularizer. It penalizes premature convergence by preventing the actor head from driving its logit parameters to extreme polarities early in training. This forces the agent to keep its options open and thoroughly explore its action space until the advantage signal provides overwhelming evidence in favor of a specific direction.
