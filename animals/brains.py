"""
ActorCriticBrain — a recurrent Actor-Critic "brain" using an LSTM memory core.

Architecture overview:
                                             ┌─────────────┐
  observation[t] ──► [Encoder] ──► embed[t] ─►             ├──► h[t]  (working memory)
                                             │    LSTM      │
                        (h[t-1], c[t-1]) ───►             ├──► c[t]  (long-term memory)
                                             └─────────────┘
                                                    │
                          ┌─────────────────────────┤  (reads h[t])
                          │                         │
                          ▼                         ▼
                    [Actor Head]              [Critic Head]
                   action logits / μ        scalar state value V(s)
                          │
                          ▼
                    action_dist
                    (sample or argmax)

LSTM vs GRU:
  GRU  hidden state : one tensor  h       shape (num_layers, batch, hidden_dim)
  LSTM hidden state : two tensors (h, c)  each  (num_layers, batch, hidden_dim)

  h = "hidden state"  — what gets read by downstream layers (actor, critic)
  c = "cell state"    — an internal memory highway; never read directly, only
                        written and passed forward. This is the LSTM's long-range
                        storage mechanism — gradients flow through c much more
                        cleanly than through h, which is why LSTMs handle long
                        episodes better than GRUs in practice.

Key design choices:
  - hidden state is a (h, c) TUPLE managed OUTSIDE the brain (caller owns it).
    Reset both tensors to zeros at episode start.
  - h[t] is what the actor and critic read. c[t] is invisible to them —
    it exists purely to give the LSTM a separate gradient highway.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical          # for discrete actions
# from torch.distributions import Normal             # swap in for continuous


# ── tuneable hyper-params ────────────────────────────────────────────────────
# NOTE: OBS_DIM is no longer a free constant -- it MUST match the environment.
# Compute it from the env like:
#     V = env.obs_size                       # number of visible cells
#     obs_dim = NUM_CELL_TYPES * V + EXTRA_FEATURES
# and pass that into ActorCriticBrain(obs_dim=..., action_dim=ACTION_DIM).
# A placeholder default is kept here only so the module imports cleanly.
NUM_CELL_TYPES = 5   # World.{BLOCKED=-2, WALL=-1, EMPTY=0, FOOD=1, ANIMAL=2} -> one-hot width 5
ACTION_DIM     = 3   # World.{FORWARD=0, LEFT=1, RIGHT=2}
EXTRA_FEATURES = ACTION_DIM + 2   # prev_action (one-hot) + prev_reward + life_force

OBS_DIM     = 97    # placeholder only -- always recompute from env, see above
EMBED_DIM   = 128   # encoder output / GRU input width
HIDDEN_DIM  = 256   # GRU hidden state size
NUM_LAYERS  = 1     # LSTM depth; start at 1, increase if animal seems forgetful
# ─────────────────────────────────────────────────────────────────────────────


def build_obs_features(
    raw_obs:    torch.Tensor,   # (batch, V) long, cell codes from World.get_observations()
    prev_action: torch.Tensor,  # (batch,) long, values in {0,1,2}; use 0 at episode start
    prev_reward: torch.Tensor,  # (batch,) float
    life_force:  torch.Tensor,  # (batch,) float, range [0, max_life_force]
    max_life_force: float = 100.0,
    num_cell_types: int = NUM_CELL_TYPES,
    action_dim: int = ACTION_DIM,
    min_cell_code: int = -2,    # value of World.BLOCKED (the lowest cell code).
                                 # one_hot index = raw_code - min_cell_code, so
                                 # this MUST stay the minimum across all cell
                                 # types, and num_cell_types MUST equal
                                 # (max_cell_code - min_cell_code + 1).
) -> torch.Tensor:
    """
    Turn the environment's raw per-step signals into the flat feature vector
    the encoder expects.

    Cell codes are shifted by -min_cell_code so the lowest code (BLOCKED=-2 by
    default) maps to one_hot index 0, giving a contiguous {0..num_cell_types-1}
    range. This avoids implying any ordering between cell types (a Linear
    layer would otherwise treat WALL < EMPTY < FOOD < ANIMAL as meaningful,
    which it isn't).

    If you add a new cell type to World, update BOTH num_cell_types (total
    distinct codes) AND min_cell_code (if the new code is lower than -1) so
    that every code in range still lands in [0, num_cell_types-1].

    prev_action/prev_reward/life_force give the LSTM the (action, outcome)
    pair from the previous step, which is what lets it learn "I went forward
    into a food-rich region and got +50" as opposed to only ever seeing the
    current local glimpse.

    Returns
    -------
    features: (batch, num_cell_types * V + action_dim + 2)
    """
    batch, V = raw_obs.shape

    cell_idx = (raw_obs - min_cell_code).long()         # shift so min code -> 0
    cell_onehot = torch.nn.functional.one_hot(cell_idx, num_classes=num_cell_types)
    cell_onehot = cell_onehot.float().reshape(batch, V * num_cell_types)

    prev_action_onehot = torch.nn.functional.one_hot(
        prev_action.long(), num_classes=action_dim
    ).float()                                           # (batch, action_dim)

    life_norm = (life_force / max_life_force).unsqueeze(-1)   # (batch, 1)
    reward_feat = prev_reward.float().unsqueeze(-1)            # (batch, 1)

    return torch.cat([cell_onehot, prev_action_onehot, reward_feat, life_norm], dim=-1)


class ActorCriticBrain(nn.Module):
    """
    The animal's entire cognitive apparatus in one module.

    Inputs  : observation tensor  (batch, obs_dim) -- this is NOT the raw
              World.get_observations() tensor. It's the output of
              build_obs_features(), i.e. one-hot cell encoding concatenated
              with [prev_action_onehot, prev_reward, life_force].
              hidden state        tuple (h, c) each (num_layers, batch, hidden_dim)
    Outputs : action distribution, value estimate, new hidden state
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        embed_dim:  int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── 1. Sensory Encoder ───────────────────────────────────────────────
        # Compresses raw sensor floats into a learned embedding.
        # Two linear layers with LayerNorm gives stable gradients early in
        # training without requiring careful weight init.
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
        )

        # ── 2. Recurrent Memory ──────────────────────────────────────────────
        # LSTM has two internal streams:
        #   h  — hidden state, read by actor/critic each step
        #   c  — cell state, a memory highway never seen by downstream layers
        # The cell state is the key advantage over GRU: gradients flow through c
        # with far less vanishing, so the animal can retain information from
        # dozens or hundreds of steps ago (e.g. "that biome was sparse last time").
        self.memory = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # (batch, seq, feature) — easier to work with
        )

        # ── 3. Actor Head ────────────────────────────────────────────────────
        # Maps hidden state → action logits.
        # We'll wrap these in a Categorical distribution so we can:
        #   - sample() for exploration
        #   - log_prob() for the policy gradient loss
        #   - entropy() as a regulariser that prevents premature certainty
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, action_dim),
            # NO softmax here — Categorical takes raw logits
        )

        # ── 4. Critic Head ───────────────────────────────────────────────────
        # Maps hidden state → scalar value estimate V(s).
        # "How much future reward do I expect from this internal state?"
        # The TD error (actual - predicted) is the training signal for everything.
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
            # No activation — value is unbounded
        )

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(
        self,
        obs:    torch.Tensor,   # (batch, obs_dim)  — current sensory snapshot
        hidden: tuple[torch.Tensor, torch.Tensor],  # (h, c) each (num_layers, batch, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        One timestep of cognition.

        Returns
        -------
        action_logits  : (batch, action_dim)  — raw scores for each action
        value          : (batch, 1)           — critic's estimate of V(s)
        new_hidden     : tuple (h, c)         — updated memory state
        """

        # Encode raw sensors → embedding
        embed = self.encoder(obs)                       # (batch, embed_dim)

        # LSTM expects (batch, seq_len, features); seq_len=1 for single-step
        embed = embed.unsqueeze(1)                          # (batch, 1, embed_dim)
        lstm_out, new_hidden = self.memory(embed, hidden)   # new_hidden is (h, c)
        lstm_out = lstm_out.squeeze(1)                      # (batch, hidden_dim)

        # Actor and critic read from lstm_out (== h[t] projected through output weights)
        action_logits = self.actor(lstm_out)            # (batch, action_dim)
        value         = self.critic(lstm_out)           # (batch, 1)

        return action_logits, value, new_hidden

    # ── Helper: build the initial hidden state ────────────────────────────────
    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns zeroed (h, c) — call this at the start of each episode.
        Both tensors are zeros: no memory yet, no long-term context yet.
        """
        zeros = lambda: torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
        return zeros(), zeros()   # (h, c)

    # ── Helper: select an action ──────────────────────────────────────────────
    @torch.no_grad()
    def act(
        self,
        obs:    torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        greedy: bool = False,
    ) -> tuple[int, torch.Tensor, tuple]:
        """
        Convenience wrapper for environment interaction (no grad needed).

        Returns
        -------
        action     : int
        value      : scalar tensor  (useful for GAE / rollout logging)
        new_hidden : updated memory state to pass in at the next timestep
        """
        logits, value, new_hidden = self.forward(obs, hidden)
        dist = Categorical(logits=logits)

        action = dist.probs.argmax(dim=-1) if greedy else dist.sample()

        return action.item(), value, new_hidden

    # ── Helper: evaluate stored actions (used during the PPO update) ──────────
    def evaluate_actions(
        self,
        obs:     torch.Tensor,   # (batch, seq_len, obs_dim) — a rollout chunk
        hidden:  tuple[torch.Tensor, torch.Tensor],  # (h, c) at START of chunk
        actions: torch.Tensor,   # (batch, seq_len) — actions that were taken
        dones:   torch.Tensor,   # (batch, seq_len) — dead bugs mask
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-run a stored rollout through the current weights, wiping LSTM 
        hidden states mid-sequence for any environment that triggered a 'done'.
        """
        batch, seq_len, _ = obs.shape
        h, c = hidden

        # 1. Encode all timesteps at once — still highly efficient
        embed = self.encoder(obs.reshape(batch * seq_len, -1))
        embed = embed.reshape(batch, seq_len, -1)

        lstm_outs = []

        # 2. Loop through time to manually manage LSTM memory
        for t in range(seq_len):
            # step_dones indicates if the bug died right before this step
            step_dones = dones[:, t]

            # Reshape to (1, batch, 1) so it broadcasts over (num_layers, batch, hidden_dim)
            mask = 1.0 - step_dones.view(1, batch, 1)

            # Wipe memory for dead bugs
            h = h * mask
            c = c * mask

            # Extract the single timestep embedding: shape (batch, 1, embed_dim)
            step_embed = embed[:, t, :].unsqueeze(1)

            # Step the LSTM forward by exactly 1 timestep
            step_out, (h, c) = self.memory(step_embed, (h, c))
            
            lstm_outs.append(step_out)

        # 3. Stitch the sequence back together along the time dimension
        lstm_out = torch.cat(lstm_outs, dim=1)  # (batch, seq_len, hidden_dim)

        # 4. Proceed with actor/critic as normal
        logits = self.actor(lstm_out)                   # (batch, seq_len, action_dim)
        values = self.critic(lstm_out)                  # (batch, seq_len, 1)

        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)              # (batch, seq_len)
        entropy   = dist.entropy().mean()               # scalar

        return log_probs, values, entropy