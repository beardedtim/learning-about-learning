# Architecture Reference: Environment Matrix Engine (`world.py`)

This document provides a low-level architectural decomposition of the `World` environment engine implemented in `world.py`. The simulation framework is designed to run multiple parallel environments vectorized on an accelerator device (`cuda`), presenting a strict Partially Observable Markov Decision Process (POMDP) to the agent's brain.

---

## 1. System Grid Coordination Matrix

The simulation coordinates global physics configurations, layout matrices, and local egocentric slices across distinct parallel simulation spaces.

### Spatial Coordinates to Egocentric Features Translation

The world isolates the true global positions of food or obstacles from the agent. The conversion maps absolute global spatial states to local sensory data arrays:

```
  [Global Grid Array (Size x Size)] ──► Raycasts / Relative Indices ──► [Egocentric Sensors]
                                                                                │
  [Agent Metabolic Life Force]       ──► Bounded Floats & State flags ──► [build_obs_features]

```

---

## 2. Low-Level Component Breakdown & Design Rationale

### A. Massively Parallel Environment Step Mechanics (`World.step`)

To fully maximize GPU tensor utilization during the rollout phase, `World` evaluates step mechanics across all parallel channels simultaneously using batched tensor matrix operations.

- **Vectorized Grid Updates:**
  Instead of looping through environment indices using Python loops, actions ($0 = \text{Forward}$, $1 = \text{Left}$, $2 = \text{Right}$) are processed as indices into transformation vectors. Global headings are updated using batched modulus trigonometry or pre-computed grid offset tensors. This allows 32 parallel worlds to simultaneously calculate movement conflicts, wall collisions, and entity interactions in single-pass CUDA kernels.
- **Non-Blocking Reward Broadcasts:**
  Rewards are compiled directly into a consolidated shape `(envs, 1)` tensor. When an agent lands on a cell flagged as `FOOD`, the engine processes an additive step matching the biome's `eating_bonus`. The calculation, boundary clipping, and variable resets execute without moving data back to host memory (CPU), preventing catastrophic synchronization bottlenecks between training phases.

### B. The POMDP Sensor Grid Array Architecture

The foundational constraint of the bug's ecosystem is that it **cannot access global coordinates $(x, y)$ or absolute map headings**. The environment must explicitly handle this abstraction layer before emitting raw observation arrays.

- **Relative Rotational Transformations:**
  If the bug has directional sensors (e.g., raycasts or a local visual window), the coordinates are rotated relative to the bug's current orientation. If the bug faces North, "Forward" samples $(y-1)$; if it faces East, "Forward" samples $(x+1)$. This mathematical transformation ensures that identical configurations of obstacles look exactly the same to the agent regardless of where it is on the global grid, turning a chaotic global space into a manageable, structured local feature landscape.
- **Sensor-Mask Padding:**
  When sensors sample cells that extend past the physical bounds of the grid, the engine assigns an explicit `BLOCKED` flag (value `-2`). This distinguishes a normal impassable boundary like a `WALL` from the true spatial termination of the grid canvas, allowing the network to internalize spatial edges implicitly through local patterns.

### C. Biome Management & Metabolic Decay Math

The environment enforces life-and-death dynamics through metabolic resource consumption rules and geographic configurations.

- **Metabolic Depletion Function:**
  Every step an agent takes drains its internal life force according to its activity profile. Standing or turning might consume standard basal metabolic rates, while a forward movement consumes a higher mechanical energy rate. This internal decay is represented as:

$$\text{life\_force}_{t} = \text{clip}(\text{life\_force}_{t-1} - \Delta_{\text{metabolic}}, \, 0, \, \text{max\_life\_force})$$

- **Automatic Vectorized Reset:**
  If $\text{life\_force}_t \le 0$, the engine flags `next_dones[i] = True`. Inside the next `.step()` sequence, the environment handles a soft-reset on that specific dead index without disrupting active trajectories in neighbor channels. It resets the bug's position to a valid starting or spawn location, replenishes food arrays according to the local `BiomeConfig`, sets the internal `life_force` back to maximum, and clears out lingering local historical logs.

### D. Biome Configuration Matrix (`BiomeConfig`)

```python
@dataclass
class BiomeConfig:
    x: int
    y: int
    width: int
    height: int
    food_refresh_rate: float
    eating_bonus: float

```

- **Dynamic Resource Generation:**
  Rather than spreading food uniformly across the map, the environment isolates sub-grids into custom biomes. The `food_refresh_rate` determines the probability per step that an empty patch of ground inside that bounding box will spawn a new piece of food.
- **Gradient Profiles:**
  By grouping different biomes together (like the `jackpot_biome` and `steady_biome` seen in the `run()` phase), the environment introduces conflicting regional incentives. Agents learn to map spatial profiles by noticing changes in local food density using nothing but their working memory core.

---

## 3. Input/Output Shape Protocol

To maintain data parity with `train.py` and `brains.py`, all tensor outputs must adhere to strict dimensional rules:

```
  Method Arguments:
    actions             -> Tensor shape: (envs, 1) [Long]

  Return Tuple Packets:
    next_obs            -> Tensor shape: (envs, 1, obs_size) [Long/Float]
                           (Squeezed down to two dimensions before feature engineering)
    rewards             -> Tensor shape: (envs, 1) [Float]
    next_dones          -> Tensor shape: (envs, 1) [Byte/Bool]

```

This strict layout guarantees that outputs pass directly into `build_obs_features` without any intermediate parsing or shape adjustments, keeping the entire pipeline clean, efficient, and running completely on the GPU.
