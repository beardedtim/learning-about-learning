from log import Log
import random
import concurrent.futures
import os

import torch.multiprocessing as tmp
from world import World,generate_initial_food, generate_walls
from simulation import Simulation
#
# DEFAULTS
#

#
# Genetic algorithm default hyperparameters
#
# GENERATIONS: number of evolutionary rounds to run.
# POP_SIZE: number of candidates evaluated each generation.
# MUTATION_RATE: strength of random variation applied during breeding.
# TRIALS_PER_EPOCH: number of separate random worlds each bug plays per fitness estimate.
GENERATIONS = 100
POP_SIZE = 1250
MUTATION_RATE = 0.075
TRIALS_PER_EPOCH = 3
MAX_ITERATIONS = 1000
DEFAULT_LIFE_FORCE = 25

tmp.set_sharing_strategy('file_system')
#
# Brokwn out so that we can run in multiprocessing
#
def evaluate_single_bug_worker(args):
    # --- Accept map_layout ---
    bug, trials, fitness_fn, map_layout = args
    total_fitness = 0
    
    for _ in range(trials):
        # --- Generate the map state sequentially ---
        walls = generate_walls(map_layout)
        food = generate_initial_food(walls=walls)
        world = World(initial_food=food, initial_walls=walls)
        
        sim = Simulation(world, bug, max_iterations=MAX_ITERATIONS, life_force=DEFAULT_LIFE_FORCE)
        results = sim.run()
        
        trial_score = fitness_fn(world, results["turns_survived"])
        total_fitness += trial_score
        
    return total_fitness / trials

class EvolutionaryTrainer:
    def __init__(self, 
                 bug_class, 
                 vision_cone, 
                 fitness_fn, 
                 generations=GENERATIONS, 
                 population_size=POP_SIZE, 
                 mutation_rate=MUTATION_RATE, 
                 trials=TRIALS_PER_EPOCH,
                 map_layout="empty",
                 name=None):
        
        self.bug_class = bug_class
        self.vision_cone = vision_cone
        self.fitness_fn = fitness_fn
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.trials = trials
        self.map_layout = map_layout
        
        self.overall_best_score = -9999999 
        self.name = name

    def train(self):
        fitness_name = self.fitness_fn.__name__.upper()
        bug_name = self.bug_class.__name__
        
        Log.info(f"--- STARTING {bug_name} EVOLUTION ({fitness_name}) {self.name} ---")
        Log.info(f"Generations: {self.generations} | Population: {self.population_size} | Trials: {self.trials}")

        # 1. Initialize Generation 0 using the provided class
        population = [self.bug_class(vision_cone=self.vision_cone) for _ in range(self.population_size)]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for gen in range(self.generations):
                # 2. Evaluate Fitness via Trials
                
                # Pack up the arguments each CPU core needs into a tuple
                worker_args = [
                    (bug, self.trials, self.fitness_fn, self.map_layout)
                    for bug in population
                ]
                
                chunksize = max(1, self.population_size // (os.cpu_count() * 4))
                fitness_scores = list(executor.map(evaluate_single_bug_worker, worker_args, chunksize=chunksize))
                    
                # Assign the calculated scores back to our main population
                for bug, score in zip(population, fitness_scores):
                    bug.fitness = score

                # 3. Sort Population (Highest fitness first)
                population.sort(key=lambda b: getattr(b, 'fitness', -999999), reverse=True)
                
                top_score = population[0].fitness
                avg_score = sum(b.fitness for b in population) / self.population_size
                
                Log.info("Generation stats", 
                    gen=gen + 1, 
                    top_score=f"{top_score:.1f}", 
                    avg_score=f"{avg_score:.2f}")
                
                if top_score > self.overall_best_score:
                    self.overall_best_score = top_score
                    Log.info(f"New Apex Bug! ({fitness_name} Score: {top_score:.1f})")

                # 4. Selection and Mutation
                num_parents = max(2, self.population_size // 10)
                parents = population[:num_parents]
                
                next_generation = []
                next_generation.extend(parents) # Elitism: The absolute best bugs survive unchanged
                
                while len(next_generation) < self.population_size:
                    # Pick two parents randomly from the elite pool
                    parent_a = random.choice(parents)
                    parent_b = random.choice(parents)
                    
                    # They breed!
                    child_bug = parent_a.spawn_child(
                        mutation_rate=self.mutation_rate, 
                        other_parent=parent_b
                    )
                    
                    next_generation.append(child_bug)
                    
                population = next_generation

        Log.info(f"\n--- {bug_name} EVOLUTION COMPLETE ---")
        return population[0] # Return the absolute best bug from the final generation


#
# Fitness Functions
#
# What does it mean to "win"?
#

def fitness_gluttony(world, turns_survived):
    """
    The 'Yo, food is good' mindset.
    Ignores how many turns it took, solely rewards the amount of food eaten.
    Pure greed - only cares about feast quantity.
    """
    return float(getattr(world, 'food_collected', 0))

def fitness_longevity(world, turns_survived):
    """
    The survivalist mindset.
    Rewards staying alive as long as possible. Food is only a means to an end.
    Pure endurance - maximize lifespan regardless of consumption.
    """
    return float(turns_survived)

def fitness_efficiency(world, turns_survived):
    """
    A hybrid mindset.
    Rewards eating food, but PENALIZES taking too long to do it. 
    Balanced approach - food matters, but speed matters more.
    """
    food = getattr(world, 'food_collected', 0)
    # 50 points per food, minus 1 point for every turn wasted
    return (food * 50.0) - turns_survived

def fitness_speed_raider(world, turns_survived):
    """
    The aggressive hunter.
    Heavily rewards early food consumption - first meal is CRITICAL.
    If no food found, penalty scales over time.
    Specializes in resource-dense environments and direct paths.
    """
    food = getattr(world, 'food_collected', 0)
    if food == 0:
        # Severe penalty for never finding food
        return -turns_survived
    # Exponential reward for eating fast (first food most valuable)
    # Each food eaten at turn T gets: 100 / sqrt(T)
    # This heavily favors finding food quickly
    return (food * 100.0) - (turns_survived * 0.5)

def fitness_sustenance(world, turns_survived):
    """
    The steady forager.
    Rewards consistent food-finding over time. The rate of food consumption matters.
    (food / turns_survived) gives consumption rate - bugs that find food regularly win.
    Penalizes both starvation AND wandering without eating.
    """
    food = getattr(world, 'food_collected', 0)
    if turns_survived == 0:
        return 0.0
    consumption_rate = food / float(turns_survived)
    # Scale by turns survived to prefer bugs that both eat AND survive
    return (consumption_rate * 1000.0) + (turns_survived * 0.1)

def fitness_balanced(world, turns_survived):
    """
    The Swiss Army knife.
    Rewards a balanced approach: need both food AND longevity equally.
    Geometric mean of (food * 50) and turns_survived.
    Great all-rounder, but not specialized in anything.
    """
    food = getattr(world, 'food_collected', 0)
    food_score = max(1, food * 50.0)  # Avoid log(0)
    survival_score = max(1, float(turns_survived))
    # Geometric mean - punishes imbalance
    return (food_score * survival_score) ** 0.5

def fitness_minimalist(world, turns_survived):
    """
    The ascetic philosopher.
    Rewards LONGEVITY while finding ANY food at all.
    Gets a baseline for surviving, bonus ONLY if food is found.
    Bugs that starve get turns_survived/2. Bugs that eat get bonus.
    """
    food = getattr(world, 'food_collected', 0)
    base_score = turns_survived / 2.0
    food_bonus = food * 30.0
    return base_score + food_bonus

def fitness_feast_or_famine(world, turns_survived):
    """
    The risk-taker's paradox.
    Extreme risk-reward: massive bonus for finding food quickly, harsh penalty for starvation.
    Only food count matters - turns are a tiebreaker.
    Winners: bugs that find ANY food (even 1) survive. Losers: bugs that find nothing.
    Creates extreme specialization pressure.
    """
    food = getattr(world, 'food_collected', 0)
    if food == 0:
        return turns_survived - 100.0  # Severe starvation penalty
    # Every food is worth a LOT - makes specialization attractive
    return food * 200.0
