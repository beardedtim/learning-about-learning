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

## Step 1: Learning the Environment

It was fun learning how NN work and how we use them. It was also cool learning about a Genetic Algorithm to
create "learned" behavior. However, it wasn't "learning" at a fast enough rate. What I want to explore now
is what it takes for a bug to learn "food is always on one side". I don't want to encode that, I don't want
it to be told "this way". I want it to _learn_ that the enviroment it is in has food in _one_ area of the map.

### The World

The world consist of an `animal`, `food`, `empty`, and `wall`.

### The Goal

This will be a success if a bug can "learn" that it needs to search the map for the area the food is in and
then it just needs to hang out in that bubble.
