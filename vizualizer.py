import pygame
import sys
import json 

from bugs import (
    MemoryBug, NeuralBug, BrainBug, 
    World, generate_walls, generate_initial_food, 
    MAX_X, MAX_Y, PLAYER_CHAR, WALL_CHAR, FOOD_CHAR,
    LIFE_FORCE 
)

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

def draw_dashboard(screen, font, bug, turns, food):
    dash_x = MAX_X * CELL_SIZE
    pygame.draw.rect(screen, DASH_BG, (dash_x, 0, DASHBOARD_WIDTH, MAX_Y * CELL_SIZE))
    pygame.draw.line(screen, HIGHLIGHT_COLOR, (dash_x, 0), (dash_x, MAX_Y * CELL_SIZE), 3)

    # --- 1. Global Stats ---
    y_offset = 20
    screen.blit(font.render("--- GOD MODE DASHBOARD ---", True, HIGHLIGHT_COLOR), (dash_x + 20, y_offset))
    screen.blit(font.render(f"Bug Type: {bug.__class__.__name__}", True, TEXT_COLOR), (dash_x + 20, y_offset + 30))
    screen.blit(font.render(f"Turn: {turns}", True, TEXT_COLOR), (dash_x + 20, y_offset + 55))
    screen.blit(font.render(f"Food Eaten: {food}", True, TEXT_COLOR), (dash_x + 20, y_offset + 80))
    
    # Turn the text red if life force drops below 30%
    life_pct = bug.life_force / bug.max_life_force
    life_color = (100, 255, 100) if life_pct > 0.3 else (255, 100, 100)
    screen.blit(font.render(f"Life Force: {bug.life_force} / {bug.max_life_force}", True, life_color), (dash_x + 20, y_offset + 105))
    
    # --- 2. Live Memory Array (CONDITIONAL) ---
    y_offset += 145 # Pushed down slightly to make room for Life Force
    if hasattr(bug, 'memory'):
        screen.blit(font.render("LIVE MEMORY SCRATCHPAD:", True, HIGHLIGHT_COLOR), (dash_x + 20, y_offset))
        y_offset += 30
        
        center_x = dash_x + 150
        pygame.draw.line(screen, (100, 100, 100), (center_x, y_offset), (center_x, y_offset + (bug.memory_size * 25)), 2)
        
        for i, val in enumerate(bug.memory):
            bar_w = int(abs(val) * 100) 
            color = (100, 255, 100) if val > 0 else (255, 100, 100)
            
            screen.blit(font.render(f"M{i}: {val:>5.2f}", True, TEXT_COLOR), (dash_x + 20, y_offset + i * 25))
            if val < 0:
                pygame.draw.rect(screen, color, (center_x - bar_w, y_offset + i * 25 + 4, bar_w, 12))
            else:
                pygame.draw.rect(screen, color, (center_x, y_offset + i * 25 + 4, bar_w, 12))

        y_offset += (bug.memory_size * 25) + 30
    else:
        screen.blit(font.render("[No Memory Array Detected]", True, (100, 100, 100)), (dash_x + 20, y_offset))
        y_offset += 50

    # --- 3. Neural Outputs (Movement Desire) ---
    screen.blit(font.render("MOVEMENT IMPULSES:", True, HIGHLIGHT_COLOR), (dash_x + 20, y_offset))
    y_offset += 30

    action_scores = getattr(bug, 'last_action_scores', [0]*8)
    best_idx = action_scores.index(max(action_scores)) if action_scores else -1

    for i, direction in enumerate(bug.directions):
        score = action_scores[i]
        color = (100, 255, 100) if i == best_idx else (150, 150, 150)
        
        screen.blit(font.render(f"{direction:<14}: {score:>5.2f}", True, color), (dash_x + 20, y_offset + (i * 20)))


def run_visualizer(filename, layout="u_trap"):
    pygame.init()
    font = pygame.font.SysFont("courier", 16, bold=True)
    
    screen_width = (MAX_X * CELL_SIZE) + DASHBOARD_WIDTH
    screen_height = MAX_Y * CELL_SIZE
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
        else:
            print(f"Unknown bug type in JSON: {bug_type}")
            sys.exit()
            
    except FileNotFoundError:
        print("Could not find the JSON file!")
        sys.exit()

    # --- NEW: Initialize the bug's life force manually for the visualizer ---
    bug.max_life_force = LIFE_FORCE
    bug.life_force = LIFE_FORCE

    if hasattr(bug, 'reset_memory'):
        bug.reset_memory()

    walls = generate_walls(layout)
    food = generate_initial_food(walls=walls)
    world = World(initial_food=food, initial_walls=walls)
    
    running = True
    turns_survived = 0
    food_eaten = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- NEW: Drain life force each turn ---
        bug.life_force -= 1

        perception = world.get_perception(**bug.vision_cone)
        next_action = bug.request_action(perception=perception)
        move_result = world.move_relative(next_action)
        turns_survived += 1
        
        if move_result == "food":
            food_eaten += 1
            # --- NEW: Restore life force on eating ---
            bug.life_force = bug.max_life_force
            
        # --- NEW: Kill the simulation if the bug starves ---
        if bug.life_force <= 0:
            print(f"Bug starved to death on turn {turns_survived}!")
            running = False

        screen.fill(BG_COLOR)
        draw_world(screen, world)
        draw_dashboard(screen, font, bug, turns_survived, food_eaten)
        
        pygame.display.flip()
        clock.tick(FPS) 
        
        if turns_survived > 1500:
            running = False

    pygame.quit()

if __name__ == "__main__":
    run_visualizer("best-neural-bug-crossover-utrap-longevity.json", layout="maze")