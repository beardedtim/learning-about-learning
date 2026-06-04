import pygame
import sys
import json 
import os
import uuid

from bugs import (
    MemoryBug, NeuralBug, BrainBug, TorchBug,
)

from world import (
    RELATIVE_DIRECTIONS, World, generate_initial_food, generate_walls, MAX_X, MAX_Y,
    PLAYER_CHAR, WALL_CHAR, FOOD_CHAR
)

from trainers import DEFAULT_LIFE_FORCE

# --- VISUAL SETTINGS ---
CELL_SIZE = 30       
DASHBOARD_WIDTH = 400  
FPS = 10             

# Colors (RGB)
BG_COLOR = (30, 30, 40)
GRID_COLOR = (45, 45, 55)
WALL_COLOR = (120, 120, 130)
FOOD_COLOR = (255, 100, 100)
BUG_COLOR = (100, 255, 100)
BUG_EYE_COLOR = (0, 0, 0)

# --- A transparent yellow for the vision flashlight ---
VISION_COLOR = (255, 255, 150, 60) 

DASH_BG = (20, 20, 25)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (255, 200, 50)

def draw_world(screen, world):
    pygame.draw.rect(screen, BG_COLOR, (0, 0, MAX_X * CELL_SIZE, MAX_Y * CELL_SIZE))

    for x in range(MAX_X):
        for y in range(MAX_Y):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

    for (x, y), entity in world.state_dict.items():
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        
        if entity == WALL_CHAR:
            pygame.draw.rect(screen, WALL_COLOR, rect)
        elif entity == FOOD_CHAR:
            center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(screen, FOOD_COLOR, center, CELL_SIZE // 3)
        elif entity == PLAYER_CHAR:
            pygame.draw.rect(screen, BUG_COLOR, rect, border_radius=4)
            fx, fy = world.player_facing
            eye_x = x * CELL_SIZE + CELL_SIZE // 2 + (fx * CELL_SIZE // 3)
            eye_y = y * CELL_SIZE + CELL_SIZE // 2 + (fy * CELL_SIZE // 3)
            pygame.draw.circle(screen, BUG_EYE_COLOR, (eye_x, eye_y), 3)

def draw_vision_cone(screen, world, bug):
    """Draws a semi-transparent overlay on the tiles the bug can currently see."""
    fx, fy = world.player_facing
    px, py = world.player_loc
    
    # 1. Primary Vectors
    f_dx, f_dy = fx, fy
    b_dx, b_dy = -fx, -fy
    l_dx, l_dy = fy, -fx
    r_dx, r_dy = -fy, fx

    # 2. Diagonal Vectors (Clamped!)
    fl_dx = max(-1, min(1, f_dx + l_dx))
    fl_dy = max(-1, min(1, f_dy + l_dy))
    fr_dx = max(-1, min(1, f_dx + r_dx))
    fr_dy = max(-1, min(1, f_dy + r_dy))
    bl_dx = max(-1, min(1, b_dx + l_dx))
    bl_dy = max(-1, min(1, b_dy + l_dy))
    br_dx = max(-1, min(1, b_dx + r_dx))
    br_dy = max(-1, min(1, b_dy + r_dy))

    vectors = {
        "forward": (f_dx, f_dy),
        "left": (l_dx, l_dy),
        "back": (b_dx, b_dy),
        "right": (r_dx, r_dy),
        "forward_left": (fl_dx, fl_dy),
        "forward_right": (fr_dx, fr_dy),
        "back_left": (bl_dx, bl_dy),
        "back_right": (br_dx, br_dy)
    }

    # Create a transparent surface for the highlight
    highlight = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    highlight.fill(VISION_COLOR)

    for direction, distance in bug.vision_cone.items():
        if distance <= 0: continue
        
        dx, dy = vectors.get(direction, (0, 0))
        if dx == 0 and dy == 0: continue
        
        for step in range(1, distance + 1):
            tx = px + (dx * step)
            ty = py + (dy * step)
            
            # Draw the highlight overlay
            screen.blit(highlight, (tx * CELL_SIZE, ty * CELL_SIZE))
            
            # Vision is blocked by walls, so stop extending this ray
            if world.state_dict.get((tx, ty)) == WALL_CHAR or tx <= 0 or tx >= MAX_X or ty <= 0 or ty >= MAX_Y:
                break

def draw_dashboard(screen, font, bug, turns, food):
    dash_x = MAX_X * CELL_SIZE
    screen_h = screen.get_height() # Dynamically grab the new height
    
    pygame.draw.rect(screen, DASH_BG, (dash_x, 0, DASHBOARD_WIDTH, screen_h))
    pygame.draw.line(screen, HIGHLIGHT_COLOR, (dash_x, 0), (dash_x, screen_h), 3)

    # --- 1. Global Stats ---
    y_offset = 20
    screen.blit(font.render("--- GOD MODE DASHBOARD ---", True, HIGHLIGHT_COLOR), (dash_x + 20, y_offset))
    screen.blit(font.render(f"Bug Type: {bug.__class__.__name__}", True, TEXT_COLOR), (dash_x + 20, y_offset + 25))
    screen.blit(font.render(f"Turn: {turns}", True, TEXT_COLOR), (dash_x + 20, y_offset + 45))
    screen.blit(font.render(f"Food Eaten: {food}", True, TEXT_COLOR), (dash_x + 20, y_offset + 65))
    
    life_pct = bug.life_force / bug.max_life_force
    life_color = (100, 255, 100) if life_pct > 0.3 else (255, 100, 100)
    screen.blit(font.render(f"Life Force: {bug.life_force} / {bug.max_life_force}", True, life_color), (dash_x + 20, y_offset + 85))
    
    # --- 2. Live Memory Array ---
    y_offset += 120 
    if hasattr(bug, 'memory'):
        screen.blit(font.render("LIVE MEMORY SCRATCHPAD:", True, HIGHLIGHT_COLOR), (dash_x + 20, y_offset))
        y_offset += 25
        
        center_x = dash_x + 150
        pygame.draw.line(screen, (100, 100, 100), (center_x, y_offset), (center_x, y_offset + (bug.memory_size * 20)), 2)
        
        for i, val in enumerate(bug.memory):
            bar_w = int(abs(val) * 100) 
            color = (100, 255, 100) if val > 0 else (255, 100, 100)
            
            screen.blit(font.render(f"M{i}: {val:>5.2f}", True, TEXT_COLOR), (dash_x + 20, y_offset + i * 20))
            if val < 0:
                pygame.draw.rect(screen, color, (center_x - bar_w, y_offset + i * 20 + 4, bar_w, 10))
            else:
                pygame.draw.rect(screen, color, (center_x, y_offset + i * 20 + 4, bar_w, 10))

        y_offset += (bug.memory_size * 20) + 20
    else:
        screen.blit(font.render("[No Memory Array Detected]", True, (100, 100, 100)), (dash_x + 20, y_offset))
        y_offset += 40

    # --- 3. Neural Outputs ---
    screen.blit(font.render("MOVEMENT IMPULSES:", True, HIGHLIGHT_COLOR), (dash_x + 20, y_offset))
    y_offset += 25

    action_scores = getattr(bug, 'last_action_scores', [0]*8)
    best_idx = action_scores.index(max(action_scores)) if action_scores else -1

    directions = getattr(bug, 'directions', RELATIVE_DIRECTIONS)
    for i, direction in enumerate(directions):
        score = action_scores[i]
        color = (100, 255, 100) if i == best_idx else (150, 150, 150)
        screen.blit(font.render(f"{direction:<14}: {score:>5.2f}", True, color), (dash_x + 20, y_offset + (i * 18)))

    # --- 4. RAW SENSOR DATA ---
    y_offset += (8 * 18) + 15
    screen.blit(font.render("RAW SENSOR DATA:", True, HIGHLIGHT_COLOR), (dash_x + 20, y_offset))
    y_offset += 25
    
    if hasattr(bug, 'last_perception'):
        for direction in directions:
            view_array = bug.last_perception.get(direction, [])
            view_str = f"[{' '.join(view_array)}]" 
            color = FOOD_COLOR if FOOD_CHAR in view_array else TEXT_COLOR
            screen.blit(font.render(f"{direction:<14}: {view_str}", True, color), (dash_x + 20, y_offset))
            y_offset += 18

def run_visualizer(filename, layout="u_trap"):
    pygame.init()
    font = pygame.font.SysFont("courier", 14, bold=True) 
    
    # --- Extend the Window ---
    BOTTOM_BAR_HEIGHT = 80
    map_width = MAX_X * CELL_SIZE
    map_height = MAX_Y * CELL_SIZE
    
    screen_width = map_width + DASHBOARD_WIDTH
    screen_height = map_height + BOTTOM_BAR_HEIGHT
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Dashboard - {layout.upper()}")
    clock = pygame.time.Clock()

    print(f"Loading champion bug from {filename}...")
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        bug_type = data.get("bug_type")
        
        if bug_type == "MemoryBug":
            bug = MemoryBug.load_from_file(filename)
        elif bug_type == "NeuralBug":
            bug = NeuralBug.load_from_file(filename)
        elif bug_type == "BrainBug":
            bug = BrainBug.load_from_file(filename)
        elif bug_type == "TorchBug":
            bug = TorchBug.load_from_file(filename)
        else:
            print(f"Unknown bug type in JSON: {bug_type}")
            sys.exit()
            
    except FileNotFoundError:
        print("Could not find the JSON file!")
        sys.exit()

    bug.max_life_force = DEFAULT_LIFE_FORCE
    bug.life_force = DEFAULT_LIFE_FORCE

    if hasattr(bug, 'reset_memory'):
        bug.reset_memory()

    walls = generate_walls(layout)
    initial_food = generate_initial_food(walls=walls, layout=layout)
    
    # Pass a copy of the food list so the original layout is preserved for restarts
    world = World(initial_food=list(initial_food), initial_walls=list(walls))
    
    running = True            
    simulation_active = True  
    turns_survived = 0
    food_eaten = 0
    
    # --- Button UI Definitions ---
    # Shifted to accommodate 3 buttons
    btn_y = map_height + 20
    restart_btn = pygame.Rect(20, btn_y, 120, 40)
    save_btn = pygame.Rect(150, btn_y, 120, 40)
    close_btn = pygame.Rect(280, btn_y, 120, 40)
    dash_x = map_width

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if close_btn.collidepoint(event.pos):
                    running = False
                
                elif save_btn.collidepoint(event.pos):
                    os.makedirs("maps", exist_ok=True)
                    map_guid = f"map_{uuid.uuid4().hex[:8]}"
                    filepath = os.path.join("maps", f"{map_guid}.json")
                    
                    map_data = {
                        "layout_name": layout,
                        "walls": list(walls),
                        "initial_food": list(initial_food)
                    }
                    with open(filepath, 'w') as f:
                        json.dump(map_data, f, indent=4)
                    print(f"Map successfully saved to: {filepath}")

                elif restart_btn.collidepoint(event.pos):
                    # --- RESTART LOGIC ---
                    # 1. Rebuild a fresh world using the original saved map data
                    world = World(initial_food=list(initial_food), initial_walls=list(walls))
                    
                    # 2. Reset the bug's biological state
                    bug.life_force = bug.max_life_force
                    if hasattr(bug, 'reset_memory'):
                        bug.reset_memory()
                        
                    # 3. Reset the simulation counters and unpause
                    turns_survived = 0
                    food_eaten = 0
                    simulation_active = True
                    print("Simulation restarted on the same map.")

        if simulation_active:
            perception = world.get_perception(**bug.vision_cone)
            next_action = bug.request_action(perception=perception)
            move_result = world.move_relative(next_action)
            
            bug.life_force -= 1
            turns_survived += 1
            
            if move_result == FOOD_CHAR:
                food_eaten += 1
                bug.life_force = bug.max_life_force
                
            if bug.life_force <= 0:
                print(f"Bug starved to death on turn {turns_survived}!")
                simulation_active = False 
            
            if turns_survived > 1500:
                print("Simulation reached max turns.")
                simulation_active = False

        screen.fill(BG_COLOR)
        draw_world(screen, world)
        
        if simulation_active:
            draw_vision_cone(screen, world, bug)
            
        draw_dashboard(screen, font, bug, turns_survived, food_eaten)
        
        # --- Render UI Buttons ---
        pygame.draw.rect(screen, (50, 100, 200), restart_btn, border_radius=5) # Blue Restart
        pygame.draw.rect(screen, (50, 150, 50), save_btn, border_radius=5)     # Green Save
        pygame.draw.rect(screen, (200, 50, 50), close_btn, border_radius=5)    # Red Close
        
        restart_text = font.render("RESTART", True, (255, 255, 255))
        save_text = font.render("SAVE MAP", True, (255, 255, 255))
        close_text = font.render("CLOSE", True, (255, 255, 255))
        
        screen.blit(restart_text, (restart_btn.centerx - restart_text.get_width()//2, restart_btn.centery - restart_text.get_height()//2))
        screen.blit(save_text, (save_btn.centerx - save_text.get_width()//2, save_btn.centery - save_text.get_height()//2))
        screen.blit(close_text, (close_btn.centerx - close_text.get_width()//2, close_btn.centery - close_text.get_height()//2))

        if not simulation_active:
            dead_text = font.render("SIMULATION ENDED", True, (255, 50, 50))
            screen.blit(dead_text, (dash_x + 20, 10))

        pygame.display.flip()
        clock.tick(FPS) 

    pygame.quit()
     
if __name__ == "__main__":
    run_visualizer("bug_saves/neural-Balanced-fitness_efficiency-dungeon.json", layout="dungeon")