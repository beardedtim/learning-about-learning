import os
import torch
from torch.func import functional_call, vmap
import time
# Import your existing modules
from world import World, FunctionalWorld
from species import GeneticColonySpecies
from checkpoint import save_champion, load_champion

# --- HYPERPARAMETERS ---
POPULATION_SIZE     = 4096  #
TOTAL_STEPS         = 2000   
GENERATIONS         = 1000   
TOURNY_SIZE         = 12
MUTATION_RATE       = 0.008  # 
MUTATION_CHANCE     = 0.05   # 
WARMSTART_RATE      = 0.05   # bigger than MUTATION_RATE
WARMSTART_CHANCE    = 0.15   # bigger than MUTATION_CHANCE
# Vision
SENSOR_RADIUS       = 5 
FRONT_FOV_RADIUS    = 5 
SIDE_FOV_RADIUS     = 2 
SPECIES_FOV_DEG     = 120 

# World Complexity
GRID_SIZE           = 32     
WALL_DENSITY        = 0.20   #
MAX_GENS_PER_MAP    = 1      # 

# Survival Mechanics
MAX_LIFE            = 100.0 
MIN_FOOD            = 100    # "Training wheels" density. About 10% of the map is food.
LIFE_DECAY          = 1.0 
LIFE_REWARD         = 35.0

MODEL_RUN           = 2 # change this to increase the counter so you can see different bug values, like a version
MODEL_BASE_FILE     = f"bug-fov{SPECIES_FOV_DEG}-radius{SENSOR_RADIUS}-front{FRONT_FOV_RADIUS}-side{SIDE_FOV_RADIUS}"
MODEL_INPUT         = f"{MODEL_BASE_FILE}-{MODEL_RUN - 1}.pt" 
MODEL_OUTPUT        = f"{MODEL_BASE_FILE}-{MODEL_RUN}.pt"

DEBUG_LOGS  = True 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# EVOLUTION (Now operating on Batched Params)
# ==========================================
def evolve_params(params, fitness_scores, tournament_size=TOURNY_SIZE, mutation_rate=MUTATION_RATE, mutation_chance=MUTATION_CHANCE):
    """
    Mutates a dictionary of batched parameters functionally.
    """
    num_bugs = fitness_scores.shape[0]

    # Draw all tournament candidates at once
    candidates = torch.randint(0, num_bugs, (num_bugs, tournament_size), device=DEVICE)
    candidate_scores = fitness_scores[candidates]
    winner_indices = candidates[torch.arange(num_bugs, device=DEVICE), torch.argmax(candidate_scores, dim=1)]
    
    # Elitism: Preserve the absolute best bug in slot 0
    champion_idx = torch.argmax(fitness_scores)
    winner_indices[0] = champion_idx

    # Mutation mask: Protect slot 0 from noise
    noise_mask = torch.ones(num_bugs, device=DEVICE)
    noise_mask[0] = 0

    new_params = {}
    for name, param in params.items():
        # Select the winning weights
        new_W = param[winner_indices]
        
        # Broadcast the mask to match the parameter dimensions
        shape = [num_bugs] + [1] * (new_W.ndim - 1)
        sparse_mask = (torch.rand_like(new_W) < mutation_chance).float()
        noise = torch.randn_like(new_W) * mutation_rate * sparse_mask * noise_mask.view(*shape)
        
        # Create the newly mutated generation AND DETACH IT
        new_params[name] = (new_W + noise).detach().clone()

    return new_params


# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def train(checkpoint_path=None):
    if DEBUG_LOGS:
        print(f"--- Initializing Functional Simulation on {DEVICE.upper()} ---")

    # 1. Setup World Constants (Using the original World class to handle static configs & food spawning)
    world = World(
        grid_size=GRID_SIZE, 
        sensor_radius=SENSOR_RADIUS, 
        front_fov_radius=FRONT_FOV_RADIUS, 
        side_fov_radius=SIDE_FOV_RADIUS, 
        fov_degrees=SPECIES_FOV_DEG, 
        min_food=MIN_FOOD, 
        max_life=MAX_LIFE,
        food_reward=LIFE_REWARD,
        life_decay=LIFE_DECAY,
        device=DEVICE
    )
    
    num_sensors = world.num_sensors

    # 2. Setup the Single-Agent Brain Template
    base_brain = GeneticColonySpecies(num_sensors=num_sensors, hidden_dim=16).to(DEVICE)
    
    # 3. Create the Batched Parameters Dictionary
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if DEBUG_LOGS: print(f"Loading champion from {checkpoint_path}...")
        # Since load_champion now returns a fully hydrated model, just overwrite base_brain
        base_brain, _ = load_champion(GeneticColonySpecies, checkpoint_path, device=DEVICE)

    # Expand the single brain's weights into a batched dictionary
    params = {
        name: param.unsqueeze(0).expand(POPULATION_SIZE, *param.shape).clone()
        for name, param in base_brain.named_parameters()
    }

    # warm start diversity: mutate everyone except slot 0
    for name in params:
        noise = torch.randn_like(params[name]) * WARMSTART_RATE
        mask = (torch.rand_like(params[name]) < WARMSTART_CHANCE).float()
        params[name][1:] += noise[1:] * mask[1:]

    # 4. Define the Pure Rollout Step for ONE Agent
    def agent_rollout_step(p, obs, mem, pos, heading, life, b_map, mask):
        # A. Brain computes action
        action, new_mem = functional_call(base_brain, p, (obs, mem))
        
        # B. Physics engine
        new_pos, new_heading, new_life, new_map, ate_food = FunctionalWorld.single_step(
            action, pos, heading, life, b_map, 
            torch.tensor([[0, -1], [1, 0], [0, 1], [-1, 0]], device=DEVICE), # forward_vectors
            LIFE_DECAY, LIFE_REWARD, MAX_LIFE
        )
        
        # C. Camera gets new frame using your precomputed tables
        new_obs = FunctionalWorld.get_single_observation(
            new_pos, new_heading, new_life, new_map, 
            world.rotated_offsets, world.los_blockers, mask, MAX_LIFE
        )
        
        return new_obs, new_mem, new_pos, new_heading, new_life, new_map, ate_food

    # 5. VMAP & Compile the Step
    if DEBUG_LOGS: print("Compiling Batched Step Kernel...")
    batched_step = vmap(agent_rollout_step, in_dims=(0, 0, 0, 0, 0, 0, 0, 0))
    compiled_step = torch.compile(batched_step)
    # ==========================================
    # THE OUTER LOOP: GENERATIONS
    # ==========================================
    with torch.inference_mode():
        for generation in range(GENERATIONS):
            gen_start_time = time.time()
            if DEBUG_LOGS: print(f"\n========== GENERATION {generation} ==========")

            # A. Ask the legacy world object to generate fresh random maps & positions
            vision_masks = [torch.ones(num_sensors) for _ in range(POPULATION_SIZE)]
            world.populate(vision_masks, wall_density=WALL_DENSITY, force_recreate=generation % MAX_GENS_PER_MAP == 0)

            # B. Extract the batched tensors from the world object for the functional loop
            maps = world.map.clone()
            positions = world.positions.clone()
            headings = world.headings.clone()
            life = world.life_force.clone()
            masks = world.masks.clone()
            
            # C. Initialize purely batched memory and starting observations
            memory = torch.zeros(POPULATION_SIZE, base_brain.hidden_dim, device=DEVICE)
            
            # (Prime the observation using a quick vmap over the starting state)
            prime_obs = vmap(FunctionalWorld.get_single_observation, in_dims=(0, 0, 0, 0, None, None, 0, None))
            obs = prime_obs(positions, headings, life, maps, world.rotated_offsets, world.los_blockers, masks, MAX_LIFE)

            fitness_scores = torch.zeros(POPULATION_SIZE, device=DEVICE)

            # ==========================================
            # THE INNER LOOP: LIFETIME STEPS
            # ==========================================
            total_food_eaten_gen = 0
            for step in range(TOTAL_STEPS):
                
                # 1. Execute the incredibly fast compiled kernel
                obs, memory, positions, headings, life, maps, ate_food = compiled_step(
                    params, obs, memory, positions, headings, life, maps, masks
                )

                # 2. Add 1 point of fitness for every bug still alive
                alive_mask = (life > 0)
                if not alive_mask.any():
                    if DEBUG_LOGS: print(f"-> Extinction Event at step {step}!")
                    break
                fitness_scores += alive_mask.float()

                # 3. Handle asynchronous food spawning (Outside the compiled kernel)
                if ate_food.any():
                    # Temporarily pass the updated tensors back to the legacy world 
                    # so it can run its built-in batched spawning logic
                    world.map = maps
                    world.positions = positions
                    world._spawn_food(ate_food)
                    maps = world.map
        
                total_food_eaten_gen += ate_food.sum().item()

            # ==========================================
            # EVOLUTION PHASE
            # ==========================================
            champion_id = torch.argmax(fitness_scores).item()
            best_score = fitness_scores[champion_id].item()
            avg_score = torch.mean(fitness_scores).item()
            std_score = torch.std(fitness_scores).item()
            median_score = torch.median(fitness_scores).item()
            survivors = (life > 0).sum().item()

            gen_time = time.time() - gen_start_time
            sps = TOTAL_STEPS / gen_time

            if DEBUG_LOGS:
                print(f"Gen {generation} / {GENERATIONS} [DONE]")
                print(f"    Average Score   : {avg_score:.1f}")
                print(f"    Champion Bug ID : {champion_id} (Score: {best_score:.1f})")
                print(f"    Median Score  : {median_score:.1f}")
                print(f"    Fitness StdDev: {std_score:.1f}")
                print(f"    Total Food Eaten: {total_food_eaten_gen}")
                print(f"    Final Survivors : {survivors} / {POPULATION_SIZE}")
                print(f"    Gen Time : {gen_time:.2f}s ({sps:.0f} steps/sec)")

            # Mutate the parameters for the next generation
            params = evolve_params(params, fitness_scores)

    # ==========================================
    # SAVE THE FINAL CHAMPION
    # ==========================================
    if DEBUG_LOGS: print(f"\n=== TRAINING COMPLETE ===")
    
    final_champ_id = torch.argmax(fitness_scores).item()
    
    # Reconstruct a standard state_dict for the base_brain using the champion's slice
    champion_state_dict = {}
    for name, param in params.items():
        champion_state_dict[name] = param[final_champ_id].clone()
        
    base_brain.load_state_dict(champion_state_dict)

    # Note: world here is just passed so save_champion has environment config
    save_champion(base_brain, fitness_scores, world, filename=MODEL_OUTPUT)
    print(f"Saved Champion to {MODEL_OUTPUT}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    checkpoint = MODEL_INPUT if os.path.exists(MODEL_INPUT) else None

    if checkpoint != None:
        print(f"Starting from checkpoint: {checkpoint}")
    else:
        print(f"No checkpoint {MODEL_INPUT} found. Starting fresh")

    train(checkpoint_path=checkpoint)