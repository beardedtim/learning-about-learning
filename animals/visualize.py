import pygame
import torch
from torch.distributions import Categorical

from world import World, WorldConfig, get_sensors, BiomeConfig
from brains import ACTION_DIM, NUM_CELL_TYPES, ActorCriticBrain, build_obs_features

# ==========================================
# ACCESSIBLE COLOR PALETTE (WCAG Compliant)
# ==========================================
SB_BG = (24, 24, 24)                 # Solid dark background
TEXT_PRIMARY = (255, 255, 255)       # Pure white
TEXT_SECONDARY = (170, 170, 170)     # Light grey
ACCENT_COLOR = (255, 193, 7)         # Amber
LIFE_HIGH = (129, 199, 132)          # Accessible Green
LIFE_LOW = (229, 115, 115)           # Accessible Red

VISION_COLORS = {
    -2: (117, 117, 117),  # Wall: Solid Grey
    -1: (45, 45, 45),     # Empty: Dark Grey
     1: (129, 199, 132),  # Food: Green
     2: (100, 181, 246)   # Animal: Light Blue
}


def render_trained_brain(cfg: WorldConfig, load_path="stage2_walk_medium.pt"):
    env = World(cfg)
    
    raw_obs = env.reset(layout="easy").squeeze(1)
    V = env.obs_size
    obs_dim = NUM_CELL_TYPES * V + ACTION_DIM + 2
    
    prev_action = torch.zeros(cfg.envs, dtype=torch.long, device=cfg.device)
    prev_reward = torch.zeros(cfg.envs, device=cfg.device)
    
    obs = build_obs_features(raw_obs, prev_action, prev_reward, env.life_force.squeeze(1), max_life_force=cfg.max_life_force)

    brain = ActorCriticBrain(obs_dim=obs_dim, action_dim=3).to(cfg.device)
    brain.load_state_dict(torch.load(load_path, map_location=cfg.device))
    brain.eval() 

    h, c = brain.init_hidden(batch_size=cfg.envs, device=cfg.device)
    
    # Initialize PyGame window
    env.render(env_idx=0, fps=15, last_action=None)
    
    pygame.font.init()

    running = True

    print(f"=== Watching Trained Brain ({load_path}) ===")

    with torch.no_grad():
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action_logits, value_ext, value_int, (new_h, new_c) = brain(obs, (h, c))
            dist = Categorical(logits=action_logits)
            actions = dist.sample() 
            
            next_obs, rewards, dones = env.step(actions.unsqueeze(1))
            
            raw_obs = next_obs.squeeze(1)
            rewards = rewards.squeeze(1)
            dones = dones.squeeze(1)

            prev_action = actions
            prev_reward = rewards
            
            obs = build_obs_features(raw_obs, prev_action, prev_reward, env.life_force.squeeze(1), max_life_force=cfg.max_life_force)

            if dones.any():
                new_h[:, dones, :] = 0.0
                new_c[:, dones, :] = 0.0
                prev_action[dones] = 0
                prev_reward[dones] = 0.0

            action_taken = actions[0].item()

            env.render(env_idx=0, fps=15, last_action=action_taken)


            h, c = new_h, new_c

    pygame.quit()

if __name__ == '__main__':
    # fertile_biome = BiomeConfig(x=20, y=4, width=5, height=5, food_refresh_rate=0.75, eating_bonus=5.0)
    # # rare_but_rich_biome = BiomeConfig(x=2, y=2, width=10, height=10, food_refresh_rate=0.01, eating_bonus=100.0)
    
    # cfg = WorldConfig(
    #     grid_size=36, 
    #     envs=1, 
    #     biomes=[fertile_biome, rare_but_rich_biome], 
    #     num_bugs=1, 
    #     min_food=10,
    #     bug_sensors=get_sensors(),
    #     device='cpu' 
    # )
    crawl_biome_left = BiomeConfig(
        x=2, y=2, width=6, height=20,
        food_refresh_rate=0.5, 
        eating_bonus=100.0, 
        max_food=4
    )

    crawl_biome_right = BiomeConfig(
        x=16, y=2, width=6, height=20,
        food_refresh_rate=0.5, 
        eating_bonus=100.0, 
        max_food=4
    )

    world_cfg_crawl = WorldConfig(
        grid_size=24,
        envs=32,
        biomes=[crawl_biome_right, crawl_biome_left],
        bug_sensors=get_sensors(),
        num_bugs=1,
        min_food=8,
        device='cpu',
    )

    print("Booting visualizer...")
    render_trained_brain(world_cfg_crawl, load_path="stage1_crawl.pt")