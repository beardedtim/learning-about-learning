import pygame
import sys
import json 

from bugs import (
    MemoryBug, NeuralBug, BrainBug, TorchBug,
)

from world import (
    World, generate_initial_food, generate_walls, MAX_X, MAX_Y,
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
    pygame.draw.rect(screen, DASH_BG, (dash_x, 0, DASHBOARD_WIDTH, MAX_Y * CELL_SIZE))
    pygame.draw.line(screen, HIGHLIGHT_COLOR, (dash_x, 0), (dash_x, MAX_Y * CELL_SIZE), 3)

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

    for i, direction in enumerate(bug.directions):
        score = action_scores[i]
        color = (100, 255, 100) if i == best_idx else (150, 150, 150)
        screen.blit(font.render(f"{direction:<14}: {score:>5.2f}", True, color), (dash_x + 20, y_offset + (i * 18)))

    # --- 4. RAW SENSOR DATA ---
    y_offset += (8 * 18) + 15
    screen.blit(font.render("RAW SENSOR DATA:", True, HIGHLIGHT_COLOR), (dash_x + 20, y_offset))
    y_offset += 25
    
    if hasattr(bug, 'last_perception'):
        for direction in bug.directions:
            view_array = bug.last_perception.get(direction, [])
            
            # Format the array into a string like "[_, _, X]" for the dashboard
            view_str = f"[{' '.join(view_array)}]" 
            
            # If the bug saw food, highlight the text red!
            color = FOOD_COLOR if FOOD_CHAR in view_array else TEXT_COLOR
            
            screen.blit(font.render(f"{direction:<14}: {view_str}", True, color), (dash_x + 20, y_offset))
            y_offset += 18

def run_visualizer(filename, layout="u_trap"):
    pygame.init()
    # Switched to a slightly smaller font so everything fits cleanly
    font = pygame.font.SysFont("courier", 14, bold=True) 
    
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
    food = generate_initial_food(walls=walls)
    world = World(initial_food=food, initial_walls=walls)
    
    running = True
    turns_survived = 0
    food_eaten = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        bug.life_force -= 1

        perception = world.get_perception(**bug.vision_cone)
        next_action = bug.request_action(perception=perception)
        move_result = world.move_relative(next_action)
        turns_survived += 1
        
        if move_result == "food":
            food_eaten += 1
            bug.life_force = bug.max_life_force
            
        if bug.life_force <= 0:
            print(f"Bug starved to death on turn {turns_survived}!")
            running = False

        screen.fill(BG_COLOR)
        draw_world(screen, world)
        
        # --- NEW: Draw the flashlight overlay ---
        draw_vision_cone(screen, world, bug)
        
        draw_dashboard(screen, font, bug, turns_survived, food_eaten)
        
        pygame.display.flip()
        clock.tick(FPS) 
        
        if turns_survived > 1500:
            running = False

    pygame.quit()

if __name__ == "__main__":
    run_visualizer("bug_saves/memory-Balanced-fitness_efficiency-dungeon.json", layout="dungeon")