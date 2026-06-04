from log import Log
import random
import concurrent.futures
import os
import copy

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
def _serialize_bug(bug):
    """Create a compact, JSON-serializable description of a bug instance.

    The worker will use this spec to reconstruct a fresh bug instance locally.
    """
    spec = {
        'class_name': bug.__class__.__name__,
        'vision_cone': getattr(bug, 'vision_cone', None),
        'payload': {}
    }

    # Per-class payloads
    cls_name = spec['class_name']
    if cls_name == 'BrainBug':
        spec['payload']['genes'] = getattr(bug, 'genes', None)

    elif cls_name in ('NeuralBug', 'MemoryBug'):
        # Extract numpy weights into plain lists to avoid large pickles
        brain = getattr(bug, 'brain', None)
        if brain is not None:
            spec['payload']['weights'] = [
                brain.W1.tolist(), brain.b1.tolist(), brain.W2.tolist(), brain.b2.tolist()
            ]
        if cls_name == 'MemoryBug':
            spec['payload']['memory_size'] = getattr(bug, 'memory_size', 4)

    elif cls_name in ('TorchBug', 'SparseTorchBug'):
        brain = getattr(bug, 'brain', None)
        if brain is not None:
            # TorchBrain/SparseTorchBrain provide to_dict() which is JSON-friendly
            try:
                spec['payload']['brain'] = brain.to_dict()
            except Exception:
                spec['payload']['brain'] = None
        spec['payload']['hidden_size'] = getattr(bug, 'hidden_size', None)
        spec['payload']['num_layers'] = getattr(bug, 'num_layers', None)
        spec['payload']['mutation_sparsity'] = getattr(bug, 'mutation_sparsity', None)

    # RandomBug/ForwardBug carry no extra data
    return spec


def _deserialize_bug(spec):
    """Reconstruct a bug instance from a spec created by _serialize_bug."""
    import bugs as bugs_module
    cls_name = spec['class_name']
    vision_cone = spec.get('vision_cone', None)
    payload = spec.get('payload', {}) or {}

    BugClass = getattr(bugs_module, cls_name)

    if cls_name == 'BrainBug':
        return BugClass(vision_cone=vision_cone, genes=payload.get('genes'))

    elif cls_name == 'NeuralBug':
        weights = payload.get('weights')
        if weights:
            import numpy as np
            numpy_weights = (
                np.array(weights[0], dtype=np.float32),
                np.array(weights[1], dtype=np.float32),
                np.array(weights[2], dtype=np.float32),
                np.array(weights[3], dtype=np.float32),
            )
            from bugs import NumpyNeuralNet
            brain = NumpyNeuralNet(input_size=17, hidden_size=12, output_size=8, weights=numpy_weights)
            return BugClass(vision_cone=vision_cone, brain=brain)
        return BugClass(vision_cone=vision_cone)

    elif cls_name == 'MemoryBug':
        weights = payload.get('weights')
        mem_size = payload.get('memory_size', 4)
        if weights:
            import numpy as np
            numpy_weights = (
                np.array(weights[0], dtype=np.float32),
                np.array(weights[1], dtype=np.float32),
                np.array(weights[2], dtype=np.float32),
                np.array(weights[3], dtype=np.float32),
            )
            from bugs import NumpyNeuralNet
            brain = NumpyNeuralNet(input_size=17 + mem_size, hidden_size=20, output_size=8 + mem_size, weights=numpy_weights)
            return BugClass(vision_cone=vision_cone, brain=brain, memory_size=mem_size)
        return BugClass(vision_cone=vision_cone, memory_size=mem_size)

    elif cls_name in ('TorchBug', 'SparseTorchBug'):
        brain_dict = payload.get('brain')
        hidden_size = payload.get('hidden_size')
        num_layers = payload.get('num_layers')
        if brain_dict:
            brain = bugs_module.TorchBrain.from_dict(brain_dict)
            return BugClass(vision_cone=vision_cone, brain=brain, hidden_size=hidden_size, num_layers=num_layers)
        return BugClass(vision_cone=vision_cone, hidden_size=hidden_size, num_layers=num_layers)

    else:
        # RandomBug, ForwardBug, or other simple classes
        return BugClass(vision_cone=vision_cone)


def evaluate_single_bug_worker(args):
    # args: (bug_spec, trials, fitness_fn, map_layout)
    bug_spec, trials, fitness_fn, map_layout = args
    total_fitness = 0.0

    # Reconstruct the bug locally to avoid sending heavy tensors across IPC
    bug = _deserialize_bug(bug_spec)

    for _ in range(trials):
        walls = generate_walls(map_layout)
        food = generate_initial_food(walls=walls, layout=map_layout)
        world = World(initial_food=food, initial_walls=walls)

        sim = Simulation(world, bug, max_iterations=MAX_ITERATIONS, life_force=DEFAULT_LIFE_FORCE)
        results = sim.run()

        # Pass the entire results dict to the fitness function
        trial_score = fitness_fn(results)
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
        
        self.apex_bug = None
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
                # We serialize each bug into a lightweight spec so we don't
                # send large torch/numpy objects through the ProcessPool.
                worker_args = [
                    (_serialize_bug(bug), self.trials, self.fitness_fn, self.map_layout)
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
                
                # Save a distinct copy of the Apex Bug
                if top_score > self.overall_best_score:
                    self.overall_best_score = top_score
                    self.apex_bug = copy.deepcopy(population[0])
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
        
        if self.apex_bug is None: 
            return population[0]
        
        return population[0] # Return the absolute best bug from the final generation


#
# Fitness Functions
#
# What does it mean to "win"?
#

def fitness_gluttony(results):
    return float(results.get('food_collected', 0))

def fitness_longevity(results):
    return float(results.get('turns_survived', 0))

def fitness_efficiency(results):
    food = results.get('food_collected', 0)
    turns = results.get('turns_survived', 0)
    return (food * 50.0) - turns

def fitness_speed_raider(results):
    """
    The aggressive hunter.
    Rewards total food consumption and penalizes total time taken.
    """
    food = results.get('food_collected', 0)
    turns = results.get('turns_survived', 0)
    if food == 0:
        return -turns
    return (food * 100.0) - (turns * 0.5)