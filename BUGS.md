# Bug Evolution Simulator

## Overview

This project is an evolutionary artificial life simulation where autonomous agents ("bugs") learn how to survive in procedurally generated environments.

Bugs must:

1. Navigate a 2D world.
2. Avoid walls and dead ends.
3. Find food before starvation.
4. Adapt to different sensory capabilities.
5. Evolve better behaviors over generations.

Unlike supervised machine learning, there is no labeled dataset.

Bugs learn entirely through:

- Random mutation
- Reproduction
- Selection pressure
- Fitness functions

---

# High-Level Architecture

```text
World
 │
 ▼
Perception
 │
 ▼
Bug Brain (if applicable)
 │
 ▼
Action
 │
 ▼
World Update
 │
 ▼
Fitness Score
 │
 ▼
Evolutionary Trainer
 │
 ▼
Next Generation
```

Each bug repeatedly executes:

```text
Look
Think
Act
Survive
Repeat
```

The trainer determines which bugs reproduce.

---

# Core Goal

The simulator answers a single question:

> What kinds of intelligence emerge when survival is the only reward?

Different brains, vision systems, maps, and fitness functions create different evolutionary pressures.

---

# World System

## Coordinate Space

```text
(0,0) = top left

Valid area:

1 <= x < MAX_X
1 <= y < MAX_Y
```

Anything outside bounds behaves as a wall.

Default map:

```python
MAX_X = 20
MAX_Y = 20
```

---

## World Contents

Each tile may contain:

```python
PLAYER_CHAR = "P"
FOOD_CHAR   = "F"
WALL_CHAR   = "X"
EMPTY_CHAR  = "_"
```

World state is stored sparsely:

```python
state_dict[(x, y)] = tile_type
```

This avoids maintaining a full 20x20 grid.

---

# Survival Model

Every bug has a starvation timer.

```python
LIFE_FORCE = 25
```

Every turn:

```python
life_force -= 1
```

When food is eaten:

```python
life_force = max_life_force
```

Therefore food is not a score.

Without food the bug dies.

---

# Movement System

Movement is relative, not absolute.

A bug thinks in:

```text
forward
back
left
right
forward_left
forward_right
back_left
back_right
```

rather than:

```text
north
south
east
west
```

---

# Vision System

## Why Vision Exists

Vision is currently the bug's only source of information of the outside world.

No bug knows:

- absolute coordinates
- food locations
- map structure

Everything comes through perception.

---

## Vision Cones

Each bug species has a sensory configuration.

# Perception Pipeline

World converts bug-facing orientation into absolute raycasts.

Example:

Bug facing:

```text
NORTH
```

Relative:

```text
forward
```

becomes:

```text
(0,-1)
```

If bug turns east:

```text
forward
```

becomes:

```text
(1,0)
```

The bug never knows this happened.

Everything stays relative.

---

# Line of Sight

The simulator performs raycasts.

Example:

```text
P _ _ F
```

returns:

```python
["_", "_", "F"]
```

Walls terminate vision.

```text
P _ X F
```

returns:

```python
["_", "X"]
```

Food behind walls is invisible.

---

# Corner-Cutting Prevention

Diagonal movement and diagonal vision are intentionally constrained.

Without this:

```text
X _
_ P
```

would allow bugs to see and move through impossible gaps.

The simulator checks adjacent tiles before permitting diagonal movement.

This creates realistic collision behavior.

---

# Bug Hierarchy

```text
BaseBug
 ├── RandomBug
 ├── ForwardBug
 ├── BrainBug
 ├── NeuralBug
 ├── MemoryBug
 ├── TorchBug
 └── DynamicBug
```

Each stage adds a new capability or representation style.

---

# RandomBug

## Purpose

A baseline with no learning and no long-term strategy.

Behavior:

```text
See food?
  Move toward it.
Otherwise:
  Pick a random direction.
```

Why it is useful:

- Establishes the minimum performance floor.
- Shows whether evolution improves behavior at all.
- Separates useful strategies from luck.

How it upgrades the next bug:

- The next step adds intentional rules on top of randomness.

---

# ForwardBug

## Purpose

A simple agent built with hand-designed instincts.

Behavior:

- Prefer food when visible.
- Keep moving forward when safe.
- Avoid walls and immediate collisions.
- Back out only when trapped.

Why it is useful:

- Provides a stable, understandable baseline.
- Demonstrates how even a few rules greatly reduce dumb mistakes.
- Shows the limits of fixed heuristics.

Upgrade from RandomBug:

- Drops chance-based wandering.
- Replaces random turns with coherent survival rules.
- Gains consistent wall avoidance and forward momentum.

---

# BrainBug

## Purpose

The first agent whose behavior is shaped by evolution rather than by fixed rules.

Core idea:

- Replace handcrafted thresholds with weighted preferences.
- Let evolution tune those weights.

Genome:

```python
food_weight
wall_weight
empty_weight
```

Decision process:

- Score each direction based on nearby food, walls, and empty space.
- Closer food boosts the score.
- Walls reduce it.
- Empty space provides a small bias.
- Choose the direction with the highest score.

Why it is useful:

- Introduces evolvable parameters.
- Keeps the model simple enough to inspect.
- Converts rule design into weight search.

Upgrade from ForwardBug:

- Moves from human-coded instincts to learned behavior.
- Lets evolution discover the right tradeoffs instead of hardcoding them.
- Still retains interpretable decision logic.

---

# NeuralBug

## Purpose

The first bug with a true neural network brain.

What changed:

- Introduces a hidden layer.
- Learns nonlinear combinations of inputs.
- Can discover richer strategies than BrainBug.

Inputs (17 total):

- 8 food proximity values
- 8 wall/contact values
- 1 normalized hunger value

Architecture:

```text
17 Inputs
   ↓
12 Hidden Neurons
   ↓
8 Outputs
```

Action selection:

```python
argmax(outputs)
```

Why it is useful:

- Learns more complex sensor-action relationships.
- Can respond differently in different contexts.
- Allows hidden representations instead of direct scoring.

Upgrade from BrainBug:

- Moves beyond a linear utility function.
- Can represent nonlinear strategies such as "avoid wall if hungry, otherwise seek food".
- Learns richer feature combinations from perception.

---

# MemoryBug

## Purpose

Give the bug a short-term memory buffer.

Problem solved:

- Feedforward networks only use the current instant.
- Navigation often requires remembering recent actions.

How it works:

- The network receives standard sensory inputs plus a small memory vector from the previous turn.
- It outputs both movement scores and new memory values.

I/O structure:

```text
Inputs:  17 sensory values + 4 memory values
Outputs: 8 movement scores + 4 new memory values
```

Why it is useful:

- Enables simple temporal reasoning.
- Helps the bug avoid repeating previous mistakes.
- Supports path-dependent strategies without a full recurrent unit.

Upgrade from NeuralBug:

- Adds state over time instead of reacting only to the current frame.
- Allows learned short-term context.
- Improves performance on dead-end and loop-heavy maps.

---

# TorchBug

## Purpose

Use a true recurrent neural network for memory.

What changed:

- Replaces manual memory feedback with a GRU.
- Hidden state flows naturally across turns.
- The brain learns temporal patterns directly.

Architecture:

```text
Inputs
  ↓
GRU
  ↓
Linear Layer
  ↓
Action Scores
```

Why it is useful:

- Supports richer sequence-based behavior.
- Avoids handcrafting memory vector mechanics.
- Internalizes temporal context in the recurrent state.

Upgrade from MemoryBug:

- Moves from explicit memory feedback to learned recurrence.
- Lets the brain decide what history is useful.
- Handles longer temporal dependencies more naturally.

---

# DynamicBug

## Purpose

Evolve the brain architecture itself, not just its weights.

What changed:

- Hidden neurons can be added or removed.
- Connections can be added, removed, or toggled.
- Both structure and weights are subject to mutation and crossover.

Why it is useful:

- Enables architecture search inside evolution.
- Lets complexity grow only when helpful.
- Can discover novel topologies for the task.

Upgrade from TorchBug:

- Adds structural evolution on top of recurrent learning.
- Lets the brain adapt its own capacity and wiring.
- Bridges fixed-model neuroevolution and topology evolution.

---

# Map Generation

Different maps create different selection pressures.

---

## Empty

```text
No walls.
```

Pure food-finding challenge.

---

## Scattered

Random obstacles.

Tests navigation.

---

## Divider

```text
█████ gap █████
```

Requires discovering passageways.

---

## U-Trap

Creates dead-end structures.

Tests escape behavior.

---

## Maze

Recursive DFS maze.

Characteristics:

```text
Single-path corridors
Many dead ends
Loops added later
```

Strong memory challenge.

---

## Dungeon

Rooms connected by corridors.

Most natural environment.

Combines:

```text
Exploration
Navigation
Decision making
```

This is currently the primary training environment.

---

# Simulation Loop

Every turn:

```python
perception = world.get_perception()

action = bug.request_action(perception)

world.move_relative(action)
```

If food eaten:

```python
life_force reset
```

If life force reaches zero:

```python
death
```

---

# Fitness Functions

The most important part of evolution.

Fitness determines what "success" means.

---

## Gluttony

```python
food_collected
```

Only food matters.

Creates aggressive foragers.

---

## Longevity

```python
turns_survived
```

Only survival matters.

Creates cautious behavior.

---

## Efficiency

```python
(food * 50) - turns
```

Rewards speed.

---

## Speed Raider

Strong preference for early food.

Produces hunters.

---

## Sustenance

Rewards food acquisition rate.

Encourages consistency.

---

## Balanced

Geometric mean:

```text
Food × Survival
```

Punishes specialization.

---

## Minimalist

Rewards survival first.

Food is bonus.

---

## Feast or Famine

Extreme specialization pressure.

Food:

```text
Huge reward
```

No food:

```text
Huge punishment
```

Creates risky strategies.

---

# Evolutionary Algorithm

## Generation Cycle

### 1. Create Population

```python
population_size = 1250
```

---

### 2. Evaluate

Each bug runs:

```python
TRIALS_PER_EPOCH
```

independent worlds.

Average fitness is used.

This reduces luck.

---

### 3. Rank

```python
population.sort(...)
```

Highest fitness first.

---

### 4. Select Parents

Top:

```python
population_size / 10
```

survive as breeding pool.

---

### 5. Reproduce

Parents create offspring via:

```text
Crossover
Mutation
```

---

### 6. Elitism

Best parents survive unchanged.

This guarantees no regression.

---

# Multiprocessing Strategy

Fitness evaluation dominates runtime.

The trainer uses:

```python
ProcessPoolExecutor
```

to distribute bug evaluations across CPU cores.

Each worker:

```text
Generate world
Run simulation
Compute fitness
Return score
```

This scales nearly linearly with available cores.

---

# Persistence

All advanced bugs support serialization.

Saved data includes:

```text
Vision configuration
Brain architecture
Weights
```

Stored as JSON.

This allows:

```text
Train once
Save forever
Load later
Compare species
Tournament testing
```

---

# Conceptual Progression

The project is a ladder of intelligence experiments:

```text
RandomBug
↓
ForwardBug
↓
BrainBug
↓
NeuralBug
↓
MemoryBug
↓
TorchBug
```

Each rung answers a different research question:

| Bug        | Question                                      |
| ---------- | --------------------------------------------- |
| RandomBug  | Is any behavior better than chance?           |
| ForwardBug | How far do instincts get us?                  |
| BrainBug   | Can simple weights evolve useful strategies?  |
| NeuralBug  | Can reactive neural intelligence emerge?      |
| MemoryBug  | Does explicit memory improve survival?        |
| TorchBug   | Can recurrent memory evolve naturally?        |
| DynamicBug | Can evolving topology discover better brains? |

---
