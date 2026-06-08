import torch

def save_champion(brain, fitness_scores, world, filename="champion.pt"):
    champion_id = torch.argmax(fitness_scores).item()
    
    torch.save({
        "champion_id": champion_id,
        "score": fitness_scores[champion_id].item(),
        "W1":    brain.W1[champion_id].cpu(),
        "b1":    brain.b1[champion_id].cpu(),
        "W2":    brain.W2[champion_id].cpu(),
        "b2":    brain.b2[champion_id].cpu(),
        "W_rec": brain.W_rec[champion_id].cpu(),
        # Save world config so we can reconstruct it exactly
        "world_config": {
            "grid_size":       world.grid_size,
            "sensor_radius":   world.sensor_radius,
            "fov_degrees":     world.fov_degrees,
            "front_fov_radius": world.front_fov_radius,
            "side_fov_radius":  world.side_fov_radius,
            "max_life":        world.MAX_LIFE,
            "food_reward":     world.FOOD_REWARD,
            "life_decay":      world.LIFE_DECAY,
            "min_food":        world.min_food,
        }
    }, filename)
    
    print(f"[Checkpoint] Saved champion {champion_id} (score {fitness_scores[champion_id].item():.1f}) to {filename}")


def load_champion(brain_class, filename="champion.pt", device="cuda"):
    data = torch.load(filename, map_location=device)
    
    input_dim, hidden_dim = data["W1"].shape
    num_sensors = input_dim - 1
    _, output_dim = data["W2"].shape

    print(f"[Checkpoint] Loading champion {data['champion_id']} (score {data['score']})")
    print(f"             num_sensors={num_sensors}, hidden_dim={hidden_dim}, output_dim={output_dim}")

    brain = brain_class(num_bugs=1, num_sensors=num_sensors, hidden_dim=hidden_dim, device=device)

    with torch.no_grad():
        brain.W1[0].copy_(data["W1"].to(device))
        brain.b1[0].copy_(data["b1"].to(device))
        brain.W2[0].copy_(data["W2"].to(device))
        brain.b2[0].copy_(data["b2"].to(device))
        brain.W_rec[0].copy_(data["W_rec"].to(device))

    return brain, data