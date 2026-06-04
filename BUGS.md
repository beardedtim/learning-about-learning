**Overview**

- **File:** [bugs.py](bugs.py)
- **Purpose:** Simulate simple agents ("bugs") in a bounded tile-based world, evolve their decision-making using evolutionary algorithms, and save trained agents.

**What `bugs.py` does**

- **World Simulation:** Implements a sparse tile map (walls, food, player) and utilities for raycast-based perception and movement.
- **Agent Types:** Provides multiple agent implementations:
  - **RandomBug / ForwardBug / BrainBug:** rule-based baselines.
  - **NeuralBug:** feed-forward neural agent (17 inputs → hidden → 8 outputs).
  - **MemoryBug:** neural agent with a small scratchpad memory appended to inputs/outputs.
- **Training Harness:** `EvolutionaryTrainer` runs populations of agents across generations using multiprocessing, selection, crossover, and mutation.
- **Fitness Functions:** Multiple fitness functions (gluttony, longevity, efficiency, speed_raider, sustenance, balanced, minimalist, feast_or_famine) to define different objective behaviors.

**Why it’s built this way**

- **Sparse state map (`state_dict`)**: keeps memory usage low and simplifies lookup for objects (food/walls/player).
- **Raycast perception**: realistic directional sensing (vision cones) that blocks on walls; feeds directly into agent inputs so agents learn to act on limited, local information.
- **Multiple agent types**: fast baselines (Random/Forward/Brain) let you compare learned behaviors against simple heuristics.
- **Simple neural architecture**: small, fast networks (NumPy-only) keep evaluations cheap and easy to serialize/deserialize.
- **Memory via scratchpad**: a simple, interpretable way to give agents short-term state without full recurrent nets.
- **Evolutionary approach**: population-based search (selection + mutation + crossover) is simple to reason about and robust for these discrete action spaces.

**How to run (local)**

- Activate your virtualenv if you use one, then run the main trainer:

```bash
source .venv/bin/activate
python bugs.py
```

- The code uses `EvolutionaryTrainer` and iterates over vision presets and fitness functions; trained agents are written to `bug_saves/`.

**Important implementation notes / gotchas**

- World coordinates: (0,0) is top-left. Valid playable tiles are inside `1 .. MAX_X-1` and `1 .. MAX_Y-1`.
- `LIFE_FORCE` is the number of turns a bug can survive without eating; eating resets it.
- Vision cone values are ray lengths passed to `get_perception()`; a value of `0` => no sight in that direction.

**Where to look in the code**

- Main simulation & world: [bugs.py](bugs.py#L300)
- Agent classes: `BrainBug`, `NeuralBug`, `MemoryBug` in [bugs.py](bugs.py#L650)
- Trainer: `EvolutionaryTrainer` in [bugs.py](bugs.py#L1200)
- Fitness functions: near the file end in [bugs.py](bugs.py#L1293)
