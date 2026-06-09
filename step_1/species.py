import torch
import torch.nn as nn
import torch.nn.functional as F


class BugActorCritic(nn.Module):
    """
    PPO Actor-Critic replacement for GeneticColonySpecies.

    The GA version was a single network that output an action (argmax over logits).
    That worked fine for evolution — you just needed a behaviour.

    PPO needs two things the GA version never had to produce:
      1. A PROBABILITY DISTRIBUTION over actions (the Actor), not just the best one.
         PPO's loss function compares the probability of the action you took NOW
         vs the probability of that same action when you first collected the rollout.
         Without probabilities, you can't compute that ratio, and the whole algorithm
         falls apart.

      2. A VALUE ESTIMATE of the current state (the Critic).
         This is a scalar: "how much total future reward do I expect from here?"
         PPO uses this to compute the Advantage (was this action better or worse
         than expected?) which dramatically reduces variance vs raw returns.

    The memory / GRU-style cell is kept because the bug still lives in a partially
    observable world and needs to remember where it has been.
    """

    def __init__(self, num_sensors, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        input_dim  = num_sensors + 1   # sensors + normalised life force (same as before)
        action_dim = 3                 # forward, turn right, turn left

        # ── Shared Encoder ────────────────────────────────────────────────────
        # WHY SHARED: Both the Actor and Critic benefit from the same
        # understanding of the world. Sharing the encoder means they build
        # a common representation of "what's happening", then specialise
        # from there. This is cheaper and usually trains faster than two
        # completely separate networks.
        #
        # WHY LARGER (64 vs 16): GA evolution tolerates tiny networks because
        # a bad individual just dies and selection pressure does the rest.
        # PPO gradient descent needs enough capacity to represent both a
        # policy AND a value function. 16 hidden units is often too tight.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        # WHY Tanh HERE instead of SiLU: The encoder output feeds into a GRU.
        # GRUs are designed around Tanh/sigmoid internals. Keeping the same
        # activation family throughout stabilises gradient flow.
        # SiLU is great for deep feedforward nets but less important here.

        # ── Recurrent Memory (GRU cell) ───────────────────────────────────────
        # WHY GRU instead of the hand-rolled gate from GeneticColonySpecies:
        # Your GA gate was a simplified GRU — one gate, one candidate.
        # A real GRUCell adds a reset gate, which lets the bug actively
        # *forget* old information when it detects a new situation.
        # That matters more in RL than GA: a GA bug that gets confused just
        # dies; a PPO bug needs to recover and keep learning from that mistake.
        #
        # GRUCell takes (input, hidden) and returns new_hidden.
        # We'll pass encoder output as the input.
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # ── Actor Head ────────────────────────────────────────────────────────
        # Outputs RAW LOGITS — one per action.
        # We do NOT argmax here. The training loop calls torch.distributions
        # .Categorical(logits=logits) to sample during rollout and to compute
        # log-probabilities during the PPO update.
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # ── Critic Head ───────────────────────────────────────────────────────
        # Outputs a SINGLE SCALAR: the estimated value V(s).
        # This head did not exist in the GA version at all.
        # During training, PPO computes:
        #   advantage = actual_return - V(s)
        #   value_loss = (actual_return - V(s))^2
        # The actor uses the advantage to know whether its action was good
        # or bad *relative to what was expected*, not just in absolute terms.
        self.critic_head = nn.Linear(hidden_dim, 1)

        # ── Weight initialisation ─────────────────────────────────────────────
        # Standard PPO practice: scale down the actor's final layer so the
        # initial policy is close to uniform (equal probability for all actions).
        # A strong initial bias toward one action can cause early collapse
        # before the bug has explored enough to know what's actually good.
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, obs, memory):
        """
        Single forward pass — used during ROLLOUT COLLECTION only.

        Called every environment step to get an action to execute.
        This is analogous to GeneticColonySpecies.forward(), but instead
        of returning a hard action we return a sampled action + its
        log-probability + the value estimate. All three are needed to
        build the rollout buffer that PPO trains on.

        Args:
            obs:    (num_envs, obs_dim)     float32
            memory: (num_envs, hidden_dim)  float32  — the bug's hidden state

        Returns:
            action:      (num_envs,)           long    — sampled action index
            log_prob:    (num_envs,)           float32 — log π(action | obs)
            value:       (num_envs,)           float32 — V(s) estimate
            new_memory:  (num_envs, hidden_dim) float32
        """
        # 1. Encode the observation into a shared feature vector
        features = self.encoder(obs)                          # (E, hidden_dim)

        # 2. Update memory with the GRU
        new_memory = self.gru_cell(features, memory)          # (E, hidden_dim)

        # 3. Actor: sample an action from the policy distribution
        logits = self.actor_head(new_memory)                  # (E, action_dim)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()                                # (E,)  ← stochastic!
        # WHY SAMPLE not ARGMAX: Exploration. During training the bug must try
        # suboptimal actions to discover if they're actually better long-term.
        # argmax always picks the current best guess and the policy never
        # escapes local optima. PPO clips the update so large deviations are
        # penalised, but you need some randomness to start with.
        log_prob = dist.log_prob(action)                      # (E,)

        # 4. Critic: estimate how good this state is
        value = self.critic_head(new_memory).squeeze(-1)      # (E,)

        return action, log_prob, value, new_memory

    def evaluate_actions(self, obs, memory, actions):
        """
        Re-score OLD actions using the CURRENT policy weights.

        This is the second forward pass that only happens during the PPO
        UPDATE step (not during rollout). It does not exist in the GA version.

        After collecting a batch of (obs, action, reward) rollout data, PPO
        needs to ask: "given what I know NOW, what log-probability would I
        assign to each action I took THEN?"

        The ratio new_log_prob / old_log_prob is the core of the PPO surrogate
        loss. If the ratio drifts too far from 1.0, the clip penalty kicks in.

        Args:
            obs:     (T*E, obs_dim)       — flattened rollout observations
            memory:  (T*E, hidden_dim)    — hidden states at each step
            actions: (T*E,)  long         — the actions that were actually taken

        Returns:
            log_prob:  (T*E,)   log π_new(action | obs)
            entropy:   (T*E,)   entropy of the distribution (used as bonus)
            value:     (T*E,)   V(s) under current critic weights
        """
        features   = self.encoder(obs)
        new_memory = self.gru_cell(features, memory)

        logits   = self.actor_head(new_memory)
        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)

        # WHY ENTROPY: PPO adds a small entropy bonus to the loss to discourage
        # the policy from collapsing to a single action too quickly.
        # High entropy = more exploratory. The coefficient (typically 0.01)
        # is a hyperparameter you tune.
        entropy = dist.entropy()

        value = self.critic_head(new_memory).squeeze(-1)

        return log_prob, entropy, value

    def init_memory(self, num_envs, device):
        """
        Returns a zeroed hidden state for a fresh episode.

        Called by the training loop on reset, just like you reset
        the memory tensor in the GA rollout loop.
        """
        return torch.zeros(num_envs, self.hidden_dim, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# What the PPO training loop looks like around this class
# (pseudo-code — not runnable, just to show how the pieces connect)
# ─────────────────────────────────────────────────────────────────────────────
#
#   bug    = BugActorCritic(num_sensors=world.num_sensors).to(device)
#   opt    = torch.optim.Adam(bug.parameters(), lr=3e-4)
#   memory = bug.init_memory(num_envs, device)
#
#   # ── Rollout collection ──────────────────────────────────────────────────
#   for step in range(rollout_steps):
#       with torch.no_grad():
#           action, log_prob, value, memory = bug(obs, memory)
#
#       obs, reward, done = world.step(action)
#
#       buffer.store(obs, action, log_prob, value, reward, done, memory)
#
#       # Reset memory for envs that just finished an episode
#       if done.any():
#           memory[done] = 0.0
#           world.reset(done.nonzero().squeeze(1))
#
#   # ── PPO update ──────────────────────────────────────────────────────────
#   for epoch in range(ppo_epochs):
#       for batch in buffer.mini_batches():
#           log_prob_new, entropy, value_new = bug.evaluate_actions(
#               batch.obs, batch.memory, batch.actions
#           )
#
#           # Advantage: was this action better or worse than expected?
#           advantage = batch.returns - batch.values
#
#           # Clipped surrogate loss (the PPO part)
#           ratio       = (log_prob_new - batch.log_probs).exp()
#           clip_ratio  = ratio.clamp(1 - clip_eps, 1 + clip_eps)
#           actor_loss  = -torch.min(ratio * advantage, clip_ratio * advantage).mean()
#
#           # Value loss: make the critic's predictions accurate
#           value_loss  = F.mse_loss(value_new, batch.returns)
#
#           # Entropy bonus: keep the policy from collapsing
#           entropy_loss = -entropy.mean()
#
#           loss = actor_loss + 0.5 * value_loss + 0.01 * entropy_loss
#           opt.zero_grad()
#           loss.backward()
#           torch.nn.utils.clip_grad_norm_(bug.parameters(), 0.5)
#           opt.step()