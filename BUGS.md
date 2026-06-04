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
 └── TorchBug
```

Each level introduces more intelligence.

---

# RandomBug

## Purpose

Baseline benchmark.

Behavior:

```text
See food?
  Move toward it.

Otherwise:
  Pick random direction.
```

Useful for answering:

> Is evolution actually helping?

---

# ForwardBug

Hand-written instinctive creature.

Strategy:

```text
Prefer food
Keep moving forward
Avoid walls
Back out of traps
```

Acts like a simple animal.

Provides a stronger baseline.

---

# BrainBug

First evolvable intelligence.

## Genome

```python
food_weight
wall_weight
empty_weight
```

Example:

```python
{
    "food_weight": 0.8,
    "wall_weight": -0.4,
    "empty_weight": 0.1
}
```

---

## Decision Process

Each direction receives a score.

Food:

```text
positive
```

Walls:

```text
negative
```

Empty space:

```text
small bias
```

Highest score wins.

---

## Evolution

Mutation alters weights:

```python
weight += random_noise
```

This is essentially a tiny evolved utility function.

---

# NeuralBug

First true neural network agent.

---

## Inputs

17 values.

### Food Signals

8 values.

One per direction.

```text
1 / distance_to_food
```

Example:

```python
0.5
```

means food is two tiles away.

---

### Wall Signals

8 values.

```python
1.0
```

means wall adjacent.

```python
0.0
```

means clear.

---

### Hunger Signal

1 value.

```python
life_force / max_life_force
```

Represents urgency.

---

## Architecture

```text
17 Inputs
   ↓
12 Hidden Neurons
   ↓
8 Outputs
```

Outputs correspond directly to movement choices.

---

## Action Selection

```python
argmax(outputs)
```

Highest score wins.

---

# MemoryBug

Problem:

Feedforward networks have no memory.

They react only to the current frame.

---

## Solution

Add a memory vector.

Input:

```text
17 sensory values
+
4 memory values
```

Output:

```text
8 movement scores
+
4 new memory values
```

The network effectively writes notes to itself.

---

## Why This Matters

MemoryBug can theoretically learn:

```text
I already explored this hallway.
```

or

```text
I turned left three turns ago.
```

without explicit programming.

This is a primitive recurrent network.

---

# TorchBug

Most sophisticated agent.

---

## Motivation

MemoryBug manually feeds memory back.

TorchBug uses a real recurrent architecture.

---

## Architecture

```text
Inputs
  ↓
GRU
  ↓
Linear Layer
  ↓
Action Scores
```

---

## GRU Hidden State

The hidden state persists between turns.

This gives the agent:

```text
Short-term memory
Context
Temporal reasoning
```

without manually managing memory vectors.

---

## Evolution Strategy

TorchBug is not trained with backpropagation.

Instead:

```text
Random initialization
Mutation
Selection
Crossover
```

All GRU weights evolve genetically.

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

| Bug        | Question                                     |
| ---------- | -------------------------------------------- |
| RandomBug  | Is any behavior better than chance?          |
| ForwardBug | How far do instincts get us?                 |
| BrainBug   | Can simple weights evolve useful strategies? |
| NeuralBug  | Can reactive neural intelligence emerge?     |
| MemoryBug  | Does explicit memory improve survival?       |
| TorchBug   | Can recurrent memory evolve naturally?       |

---
