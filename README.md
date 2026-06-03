# Learning About Learning

## Overview

The goal of this project is to learn about how computer's _learn_
by trying to create "learning" things to solve "problems" in the "real"
world.

## Step 0: Bugs

The first "thinking" thing I am going to learn about is a random "bug",
or some "thing" inside of a 2d grid that needs to "eat" food to survive.
It will have some "perception" of the "world" and be able to make a choice
in what to do each "turn".

### World

The world will have empty space, the `bug`, and `food` randomly thrown through

### Goal

The goal is to survive/continually eat food for the longest iterations.

### Running

You can run the different trials via `bugs.py`:

```python
if __name__ == "__main__":
    basic_bugs() # tests basic bugs to get a baseline
    train_genetic_algorithm() # tests genetic algorithm
    train_neural_algorithm() # tests a basic single neural net
    train_neural_algorithm_with_trials() # tests neural net with many trials
    train_neural_algorithm_with_trials_and_fitness(fitness_fn=fitness_efficiency) # tests neural net with many trials and a custom fitness function
```
