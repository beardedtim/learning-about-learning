import os

from bugs import VISION_CONES, MemoryBug, NeuralBug, TorchBug
from trainers import EvolutionaryTrainer, fitness_balanced, fitness_efficiency, fitness_feast_or_famine, fitness_gluttony, fitness_longevity, fitness_minimalist, fitness_speed_raider, fitness_sustenance

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

if __name__ == "__main__":
    #
    # Training Config
    #
    fitness_fns = [
        fitness_efficiency, 
        fitness_gluttony, 
        fitness_longevity,
        fitness_speed_raider,
        fitness_sustenance,
        fitness_balanced,
        fitness_minimalist,
        fitness_feast_or_famine
    ]

    MAP_CREATION_TYPE = "dungeon"

    for fitness_fn in fitness_fns:
        for name, vision in VISION_CONES.items():
            print("-"*10)
            print(f"Vision: {name} - {fitness_fn.__name__}")
            print("-"*10)

            neural_path  = f'bug_saves/neural-{name}-{fitness_fn.__name__}-dungeon.json'
            memory_path  = f"bug_saves/memory-{name}-{fitness_fn.__name__}-dungeon.json"
            torchnn_path = f"bug_saves/torchnn-{name}-{fitness_fn.__name__}-dungeon.json"

            if not os.path.exists(neural_path):
                trainer_neural = EvolutionaryTrainer(
                    bug_class=NeuralBug,
                    vision_cone=vision,
                    generations=GENERATIONS,
                    population_size=POP_SIZE,
                    mutation_rate=MUTATION_RATE,
                    fitness_fn=fitness_fn,
                    map_layout=MAP_CREATION_TYPE,
                    trials=TRIALS_PER_EPOCH,
                    name="Neural"
                )
                best_neural = trainer_neural.train()
                best_neural.save_to_file(neural_path)

            if not os.path.exists(memory_path):
                trainer_memory = EvolutionaryTrainer(
                    bug_class=MemoryBug,
                    vision_cone=vision,
                    fitness_fn=fitness_fn,
                    generations=GENERATIONS,
                    population_size=POP_SIZE,
                    mutation_rate=MUTATION_RATE,
                    map_layout=MAP_CREATION_TYPE,
                    trials=TRIALS_PER_EPOCH,
                    name="Memory"
                )
                best_memory = trainer_memory.train()
                best_memory.save_to_file(memory_path)

            if not os.path.exists(torchnn_path):
                trainer_torchnn = EvolutionaryTrainer(
                    bug_class=TorchBug,
                    vision_cone=vision,
                    fitness_fn=fitness_fn,
                    generations=GENERATIONS,
                    population_size=POP_SIZE,
                    mutation_rate=MUTATION_RATE,
                    map_layout=MAP_CREATION_TYPE,
                    trials=TRIALS_PER_EPOCH,
                    name="TorchNN"
                )
                best_torchnn = trainer_torchnn.train()
                best_torchnn.save_to_file(torchnn_path)

            print("")
            print("")