import pygame
import torch
from torch.distributions import Categorical

from train import crawl_ppo_training_config
from world import World, WorldConfig
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


def render_trained_brain(cfg: WorldConfig, load_path="stage2_walk_medium.pt", layout = "easy"):
    env = World(cfg)
    
    raw_obs = env.reset(layout=layout).squeeze(1)
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

            env.render(env_idx=0, fps=15, last_action=action_taken, layout=layout, brain_state=current_brain_state)


            h, c = new_h, new_c

    pygame.quit()

if __name__ == '__main__':
    # Crawl Config
    (world_cfg, _) = crawl_ppo_training_config()

    print("Booting visualizer...")
    render_trained_brain(world_cfg, load_path="stage1_crawl.pt", layout="easy")