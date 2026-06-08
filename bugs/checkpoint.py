import torch

def save_champion(brain, fitness_scores, world, filename):
    """
    Saves the champion's standard state_dict and the world configuration.
    """
    world_config = {
        "grid_size": world.grid_size,
        "sensor_radius": world.sensor_radius,
        "fov_degrees": world.fov_degrees,
        "front_fov_radius": world.front_fov_radius,
        "side_fov_radius": world.side_fov_radius,
        "max_life": world.MAX_LIFE,
        "food_reward": world.FOOD_REWARD,
        "life_decay": world.LIFE_DECAY,
        "min_food": world.min_food,
    }

    # We now use PyTorch's built-in state_dict() to save all weights effortlessly
    checkpoint = {
        "model_state_dict": brain.state_dict(),
        "world_config": world_config,
        "best_score": torch.max(fitness_scores).item() if fitness_scores is not None else 0.0
    }
    
    torch.save(checkpoint, filename)

def load_champion(model_class, filename, device="cuda"):
    """
    Instantiates the model, loads the weights, and returns both the model and config.
    """
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    
    # Auto-detect the neural network dimensions directly from the saved weights!
    hidden_dim, input_dim = state_dict["vision_enc.weight"].shape
    num_sensors = input_dim - 1
    
    # Reconstruct and hydrate the champion brain
    brain = model_class(num_sensors=num_sensors, hidden_dim=hidden_dim).to(device)
    brain.load_state_dict(state_dict)
    
    return brain, checkpoint