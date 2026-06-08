import torch

from world import World
from species import GeneticColonySpecies
from checkpoint import save_champion, load_champion

POPULATION_SIZE = 20000 # How many total species are alive at any given time?
TOTAL_STEPS     = 2000 # How many steps per generation do we allow?
GENERATIONS     = 100 # How many generations do we evolve?
TOURNY_SIZE     = 4 # How much selective pressure we apply during evolution; i.e. how much it favors the best bugs
MUTATION_RATE   = 0.05 # When we mutate, what is the rate of change? 0.02 = 2%

SENSOR_RADIUS       = 5 # At the very max, how large is any individual species' sensor radius?
FRONT_FOV_RADIUS    = 5 # How far in front can it see?
SIDE_FOV_RADIUS     = 2 # How off to the side can it see?
SPECIES_FOV_DEG     = 120 # What is the sensor cone's radius of the entity?

GRID_SIZE       = 24 # XxX grid. What is X?
MAX_LIFE        = 100.0 # What is the max life any species can have?
MIN_FOOD        = 15 # Howmuch food does the world spawn with?
LIFE_DECAY      = 1.0 # How much does each species lose per turn?
LIFE_REWARD     = 100.0 # How much do they get when they eat?
MODEL_OUTPUT    = "bug.pt"

DEBUG_LOGS  = True # Do we add all the logs that make you feel good
DEVICE      = "cuda" # What device (cpu/cuda) do you want to run all this on. Tested on GPU

def evolve_population(brain, fitness_scores, tournament_size=TOURNY_SIZE, mutation_rate=MUTATION_RATE):
    num_bugs = brain.num_bugs

    # Draw all tournament candidates at once: (num_bugs, tournament_size)
    candidates = torch.randint(0, num_bugs, (num_bugs, tournament_size), device=brain.device)
    candidate_scores = fitness_scores[candidates]
    winner_indices = candidates[torch.arange(num_bugs, device=brain.device), 
                                torch.argmax(candidate_scores, dim=1)]
    
    # Preserve the elite bug in slot 0
    winner_indices[0] = torch.argmax(fitness_scores)

    # Gather all weights at once using index_select
    new_W1   = brain.W1[winner_indices]
    new_b1   = brain.b1[winner_indices]
    new_W2   = brain.W2[winner_indices]
    new_b2   = brain.b2[winner_indices]
    new_W_rec = brain.W_rec[winner_indices]

    # Mutation mask: protect slot 0
    noise_mask = torch.ones(num_bugs, device=brain.device)
    noise_mask[0] = 0

    with torch.no_grad():
        brain.W1.copy_(new_W1    + torch.randn_like(new_W1)    * mutation_rate * noise_mask.view(-1, 1, 1))
        brain.b1.copy_(new_b1    + torch.randn_like(new_b1)    * mutation_rate * noise_mask.view(-1, 1, 1))
        brain.W2.copy_(new_W2    + torch.randn_like(new_W2)    * mutation_rate * noise_mask.view(-1, 1, 1))
        brain.b2.copy_(new_b2    + torch.randn_like(new_b2)    * mutation_rate * noise_mask.view(-1, 1, 1))
        brain.W_rec.copy_(new_W_rec + torch.randn_like(new_W_rec) * mutation_rate * noise_mask.view(-1, 1, 1))

def print_generation_stats(fitness, step, total_population):
    """
    Prints a clean summary of the colony's current fitness.
    """    
    # Calculate stats
    avg_fit = torch.mean(fitness).item()
    max_fit = torch.max(fitness).item()
    min_fit = torch.min(fitness).item()
    
    # Calculate survival (how many bugs have health > 0)
    survivors = (fitness > 0).sum().item()
    survival_rate = (survivors / total_population) * 100
    
    # Find the index of the absolute best bug
    best_bug_idx = torch.argmax(fitness).item()

    print(f"--- Step {step} ---")
    print(f"Survivors : {survivors}/{total_population} ({survival_rate:.1f}%)")
    print(f"Average   : {avg_fit:.1f}")
    print(f"Worst Bug : {min_fit:.1f}")
    print(f"Best Bug  : {max_fit:.1f} (Bug ID: {best_bug_idx})")
    print("-" * 19)

def standard_survival_fitness(fitness_scores, world, actions, actually_moved):
    # One point per step alive. That's it.
    # Eating is not rewarded directly — it's just the mechanism to stay alive longer.
    alive = (world.life_force > 0).float()
    return fitness_scores + alive

def train(
        pop_size=POPULATION_SIZE, 
        sensor_radius=SENSOR_RADIUS,
        fov_deg=SPECIES_FOV_DEG,
        front_fov_radius = FRONT_FOV_RADIUS,
        side_fov_radius = SIDE_FOV_RADIUS, 
        total_steps=TOTAL_STEPS, 
        num_generations=GENERATIONS,
        min_food=MIN_FOOD,
        grid_size=GRID_SIZE,
        max_life=MAX_LIFE,
        life_decay=LIFE_DECAY,
        model_output=MODEL_OUTPUT,
        food_reward=LIFE_REWARD,
        fitness_fn=standard_survival_fitness,
        checkpoint_path=None,
        debug_logs=DEBUG_LOGS):
    # 1. INITIAL SETUP
    world = World(
        grid_size=grid_size, 
        sensor_radius=sensor_radius, 
        front_fov_radius=front_fov_radius, 
        side_fov_radius=side_fov_radius, 
        fov_degrees=fov_deg, 
        min_food=min_food, 
        max_life=max_life,
        food_reward=food_reward,
        life_decay=life_decay,
        device=DEVICE)
    
    num_sensors = world.num_sensors

    # Create the vision masks (all 1s for full vision for now)
    vision_masks = [torch.ones(num_sensors) for _ in range(pop_size)]
    
    # Initialize the World and the Batched Brain
    world.populate(vision_masks)
    brain = GeneticColonySpecies(pop_size, num_sensors, hidden_dim=16, device=DEVICE)
    if checkpoint_path is not None:
        if debug_logs == True:
            print(f"[Train] Seeding population from {checkpoint_path}...")
        
        _, data = load_champion(GeneticColonySpecies, checkpoint_path, device=DEVICE)

        champion_W1   = data["W1"].to(DEVICE)
        champion_b1   = data["b1"].to(DEVICE)
        champion_W2   = data["W2"].to(DEVICE)
        champion_b2   = data["b2"].to(DEVICE)
        champion_W_rec = data["W_rec"].to(DEVICE)

        # Broadcast champion weights into the entire population
        # They'll diverge through mutation — this just gives everyone
        # a head start from a known-good solution
        with torch.no_grad():
            brain.W1.copy_(champion_W1.unsqueeze(0).expand(pop_size, -1, -1))
            brain.b1.copy_(champion_b1.unsqueeze(0).expand(pop_size, -1, -1))
            brain.W2.copy_(champion_W2.unsqueeze(0).expand(pop_size, -1, -1))
            brain.b2.copy_(champion_b2.unsqueeze(0).expand(pop_size, -1, -1))
            brain.W_rec.copy_(champion_W_rec.unsqueeze(0).expand(pop_size, -1, -1))

        if debug_logs == True:
            print(f"[Train] Population seeded. Mutation will diversify from here.")

    # ==========================================
    # THE OUTER LOOP: GENERATIONS
    # ==========================================
    for generation in range(num_generations):
        if debug_logs == True:
            print(f"\n========== GENERATION {generation} ==========")

        # 2. GENERATION RESET
        # Create a fresh scoreboard for this generation
        fitness_scores = torch.zeros(pop_size, device=DEVICE)
        observations = world.get_observations()
        brain.reset_memory()

        # ==========================================
        # THE INNER LOOP: LIFETIME STEPS
        # ==========================================
        for step in range(total_steps):
            # Ask brains for decisions
            actions = brain(observations)
            prev_positions = world.positions.clone()
            
            # Advance environments
            observations = world.step(actions)
            actually_moved = (world.positions[:, 0] != prev_positions[:, 0]) | \
                             (world.positions[:, 1] != prev_positions[:, 1])
            
            fitness_scores = fitness_fn(fitness_scores, world, actions, actually_moved)            
            # Optional: Print mid-generation update
            # if step > 0 and step % 100 == 0:
            #     print_generation_stats(fitness_scores, step, pop_size)
            
        # ==========================================
        # EVOLUTION PHASE
        # ==========================================
        if debug_logs == True:
            print(f"\n=== GENERATION {generation} COMPLETE ===")

        # 3. Find the True Champion using their accumulated lifetime score!
        champion_id = torch.argmax(fitness_scores).item()
        best_score = fitness_scores[champion_id].item()
        avg_score = torch.mean(fitness_scores).item()
        
        if debug_logs == True:
            print(f"Average Score : {avg_score:.1f}")
            print(f"Champion Bug ID       : {champion_id} (Score {best_score})")

        # 4. Mutate and Breed the Brains
        if debug_logs == True:
            print("Evolving population...")

        evolve_population(brain, fitness_scores)

        # 5. Reset the Physical World for the next generation
        # This gives everyone a fresh mMMIN_FOODIN_FOODap, max health, and new food.
        world.populate(vision_masks)
    
    if debug_logs == True:
        print(f"\n=== TRAINING COMPLETE ===")
    

    # 3. Find the Final Champion
    champion_id = torch.argmax(fitness_scores).item()
    if debug_logs == True:
        best_score = fitness_scores[champion_id].item()
        print(f"Final Champion Bug ID : {champion_id} (Survived {best_score} steps)")
        print(f"Champion weights norm: {brain.W1[champion_id].norm():.4f}")
    
    save_champion(brain, fitness_scores, world, filename=model_output)
    print(f"Saved to {model_output}")

if __name__ == "__main__":
    import os
    checkpoint = MODEL_OUTPUT if os.path.exists(MODEL_OUTPUT) else None
    train(checkpoint_path=checkpoint)