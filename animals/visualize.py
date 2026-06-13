import pygame
import torch
from torch.distributions import Categorical

from train import crawl_ppo_training_config, walk_ppo_training_config
from world import World, WorldConfig
from brains import ACTION_DIM, NUM_CELL_TYPES, ActorCriticBrain, ActorCriticBrainConfig, build_obs_features

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


def render_trained_brain(cfg: WorldConfig, brain_cfg: ActorCriticBrainConfig, load_path="stage2_walk_medium.pt", layout = "easy"):
    cfg.num_bugs = 1
    env = World(cfg)
    
    raw_obs = env.reset(layout=layout).squeeze(1)
    V = env.obs_size
    obs_dim = NUM_CELL_TYPES * V + ACTION_DIM + 2
    
    prev_action = torch.zeros(cfg.envs, dtype=torch.long, device=cfg.device)
    prev_reward = torch.zeros(cfg.envs, device=cfg.device)
    
    obs = build_obs_features(raw_obs, prev_action, prev_reward, env.life_force.squeeze(1), max_life_force=cfg.max_life_force)

    brain_cfg.obs_dim=obs_dim

    brain = ActorCriticBrain(brain_cfg).to(cfg.device)
    brain.load_state_dict(torch.load(load_path, map_location=cfg.device))
    brain.eval() 

    h, c = brain.init_hidden(batch_size=cfg.envs, device=cfg.device)
    
    # Initialize PyGame window
    env.render(env_idx=0, fps=15, last_action=None)
    
    pygame.font.init()

    running = True

    print(f"=== Watching Trained Brain ({load_path}) ===")



    # MAX ACROSS RESETS
    max_food_so_far = 0
    total_deaths = 0

    with torch.no_grad():
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            with torch.no_grad():                
                action_logits, value_ext, value_int, (new_h, new_c) = brain(obs, (h, c))
                dist = Categorical(logits=action_logits)
                actions = dist.sample()
                # Calculate RND novelty exactly as you do in Phase 1 of training
                target_feat = brain.rnd_target(obs)
                pred_feat = brain.rnd_predictor(obs)
                novelty = torch.mean((pred_feat - target_feat)**2, dim=-1)

            # Convert logits to actual 0-100% probabilities for the UI
            action_probs = torch.softmax(action_logits, dim=-1)

            # Package the brain state for Environment 0 (the one being rendered)
            current_brain_state = {
                'probs': action_probs[0].cpu().numpy().tolist(),
                'v_ext': value_ext[0].item(),
                'v_int': value_int[0].item(),
                'novelty': novelty[0].item()
            }
            
            next_obs, rewards, dones = env.step(actions.unsqueeze(1))
            
            if dones[0]:
                env.reset()
                total_deaths += 1
                continue
            else:
                max_food_so_far = max(env.food_eaten[0].item(), max_food_so_far)

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

            env.render(env_idx=0, fps=8, last_action=action_taken, layout=layout, brain_state=current_brain_state, ctx={
                "max_food": max_food_so_far,
                "total_deaths": total_deaths
            })
    
            h, c = new_h, new_c
    pygame.quit()

if __name__ == '__main__':
    # Crawl Config
    (world_cfg, _, brain_cfg) = crawl_ppo_training_config()
    file = "stage1_crawl.pt"
    layout = "easy"

    # Walk
    # (world_cfg, _, brain_cfg) = walk_ppo_training_config()
    # file = "stage2_walk_medium.pt"
    # layout = "medium"

    print("Booting visualizer...")
    render_trained_brain(world_cfg, brain_cfg=brain_cfg, load_path=file, layout=layout)