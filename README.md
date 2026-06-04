# Learning About Learning

## Overview

The goal of this project is to learn about how computers _learn_ by creating "thinking" algorithms to solve "problems" in a simulated environment.

## Step 0: Bugs

The first "thinking" entity we explore is a "bug" placed inside a 2D grid. The bug must "eat" food to survive. It has a specific "perception" of the world (its vision cone) and makes a choice on what to do each "turn" using its internal brain.

### The World

The world consists of empty space, boundaries (walls), the `bug`, and `food` randomly scattered throughout.

### The Goal

The primary objective is survival, but "winning" can be defined in multiple ways using **Fitness Functions**:

- **Gluttony:** Eat as much food as possible, regardless of the turns taken.
- **Longevity:** Survive as many turns as possible.
- **Efficiency:** Eat food quickly, penalizing wasted turns.

---

## Local Usage & API

The simulation uses a unified `EvolutionaryTrainer` API, making it incredibly simple to test different bug architectures (`BrainBug`, `NeuralBug`, `DynamicBug`), vision cones, and fitness functions locally.

### 1. Training a New Population

You can configure and run a training session by utilizing the API directly in `bugs.py`:

```python
from bugs import EvolutionaryTrainer, NeuralBug, VISION_CONES, fitness_efficiency

if __name__ == "__main__":
    # Configure the training environment
    trainer = EvolutionaryTrainer(
        bug_class=NeuralBug,
        vision_cone=VISION_CONES["Radar"], # 360-degree vision
        fitness_fn=fitness_efficiency,     # Reward fast eaters
        generations=50,
        population_size=1000,
        trials=3                           # Average the fitness across 3 random maps
    )

    # Run the evolution and extract the absolute best bug from the final generation
    champion_bug = trainer.train()
```

### 2. Saving and Loading Champions

Once a bug has evolved a highly successful brain, you can save it to your local file system as a JSON file. This saves both its "eyes" (vision cone) and its "brain" (neural weights), allowing you to preserve your AI's hard work.

#### Saving a Bug:

```python
# Save the champion to your hard drive after training completes
champion_bug.save_to_file("apex_efficiency_radar_bug.json")
```

#### Loading a Bug:

```python
# Instantly resurrect a trained bug from disk in a completely new script
my_champion = NeuralBug.load_from_file("apex_efficiency_radar_bug.json")

# You can now put it directly into a new World and run a Simulation!
```

### 3. Logs and History

As your AI trains, all generation stats and new "Apex Bug" discoveries are automatically printed to your console and appended to a local `simulation_history.log` file. This provides a permanent, structured record of your experiments so you can compare mutation rates and fitness scores over time.
