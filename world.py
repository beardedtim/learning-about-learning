import random
import os
import json
#
# DEFAULTS
#

# Coordinate system and world bounds:
# - (0, 0) is the top-left corner of the map.
# - Valid tile coordinates are inside the range 1 .. MAX_X-1 and 1 .. MAX_Y-1.
# - Any point with x <= 0, x >= MAX_X, y <= 0, or y >= MAX_Y is treated as
#   outside the playable area and behaves like a wall.
MAX_X = 20
MAX_Y = 20

# The player's spawn location inside the world.
PLAYER_START = (10, 10)

# Default number of food items placed into each new world.
DEFAULT_INITIAL_FOOD_COUNT = 12

def generate_initial_food(walls=None, num_items=DEFAULT_INITIAL_FOOD_COUNT, layout="empty"):
    """
    Randomly generates a list of unique food coordinates, or loads them from a saved map.
    """
    # --- Check for saved map JSON ---
    if layout.startswith("map_"):
        filepath = os.path.join("maps", f"{layout}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                map_data = json.load(f)
                
            # Cast the JSON arrays back into Python tuples
            return [tuple(coord) for coord in map_data.get("initial_food", [])]

    # --- Standard Random Generation ---
    food_positions = set()
    wall_set = set(walls) if walls else set()
    
    while len(food_positions) < num_items:
        # We calculate available spaces to prevent infinite loops on densely packed maps
        available_spaces = [
            (x, y) for x in range(1, MAX_X) for y in range(1, MAX_Y)
            if (x, y) != PLAYER_START and (x, y) not in wall_set
        ]
        
        # If the map is so full of walls that we can't fit the requested food, stop early
        if len(available_spaces) < num_items:
            num_items = len(available_spaces)
            if num_items == 0:
                break
                
        new_pos = random.choice(available_spaces)
        food_positions.add(new_pos)
            
    return list(food_positions)

def generate_walls(layout="empty"):
    """Generates a list of coordinates for wall placements based on a layout preset."""
    
    # --- NEW: Check for saved map JSON ---
    if layout.startswith("map_"):
        filepath = os.path.join("maps", f"{layout}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                map_data = json.load(f)
                
            # JSON saves coordinate tuples as lists (e.g., [[1, 2], [3, 4]]). 
            # We must cast them back to tuples so they can be hashed in World's state_dict.
            return [tuple(coord) for coord in map_data.get("walls", [])]
        else:
            print(f"Warning: Saved map '{filepath}' not found. Defaulting to empty walls.")
            return []

    walls = set()
    
    if layout == "empty":
        return []
        
    elif layout == "scattered":
        # Drop 20 random blocks around the map
        for _ in range(20):
            x = random.randint(2, MAX_X - 2)
            y = random.randint(2, MAX_Y - 2)
            if (x, y) != PLAYER_START:
                walls.add((x, y))
                
    elif layout == "divider":
        # Draw a solid wall down the middle of the map with a gap in the center
        mid_x = MAX_X // 2
        gap_start = (MAX_Y // 2) - 2
        gap_end = (MAX_Y // 2) + 2
        
        for y in range(1, MAX_Y):
            # If we are NOT in the gap, place a wall
            if not (gap_start <= y <= gap_end):
                walls.add((mid_x, y))

    elif layout == "u_trap":
        # Creates a large U-shape holding pit in the center of the map.
        mid_x = MAX_X // 2
        mid_y = MAX_Y // 2

        # Bottom of the U
        for x in range(mid_x - 4, mid_x + 5):
            walls.add((x, mid_y + 2))

        # Left and Right walls of the U
        for y in range(mid_y - 3, mid_y + 3):
            walls.add((mid_x - 4, y))
            walls.add((mid_x + 4, y))
            
    elif layout == "maze":
        # 1. Fill the entire inner grid with walls
        for x in range(1, MAX_X):
            for y in range(1, MAX_Y):
                walls.add((x, y))

        # 2. Carve a perfect maze using Recursive Backtracking (DFS)
        # We step by 2 so we always leave a 1-tile thick wall between our paths
        start_x, start_y = 1, 1
        walls.discard((start_x, start_y))
        stack = [(start_x, start_y)]

        while stack:
            cx, cy = stack[-1]
            
            # Find all valid neighbors that are 2 steps away AND are still solid walls
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                # Check if the target is within bounds and uncarved
                if 0 < nx < MAX_X and 0 < ny < MAX_Y and (nx, ny) in walls:
                    neighbors.append((nx, ny, dx, dy))

            if neighbors:
                # Pick a random valid direction to dig
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Carve the destination cell
                walls.discard((nx, ny))
                
                # Carve the wall *between* the current cell and the destination
                walls.discard((cx + dx // 2, cy + dy // 2))
                
                # Move to the new cell
                stack.append((nx, ny))
            else:
                # We hit a dead end! Backtrack to the previous intersection
                stack.pop()

        # 3. Rescue the Player!
        # Because the algorithm carves on ODD coordinates (1, 3, 5), EVEN coordinates 
        # like the default PLAYER_START (10, 10) are solid permanent pillars. 
        # We must manually carve out the player's spawn and connect them to the maze.
        walls.discard(PLAYER_START)
        
        px, py = PLAYER_START
        if px % 2 == 0: 
            walls.discard((px - 1, py)) # Dig a path to the left
        if py % 2 == 0: 
            walls.discard((px, py - 1)) # Dig a path up
            
        # 4. The "Mercy" Mechanic (Braid Maze)
        # A "perfect" maze has exactly ONE path between any two points and zero loops.
        # This is incredibly brutal for AI. We punch a few random holes in the walls 
        # to create loops and shortcuts so they don't get completely trapped in dead ends.
        for _ in range(15):
            rx = random.randint(2, MAX_X - 2)
            ry = random.randint(2, MAX_Y - 2)
            walls.discard((rx, ry))
    
    elif layout == "dungeon":
        # 1. Fill the map with solid stone
        for x in range(1, MAX_X):
            for y in range(1, MAX_Y):
                walls.add((x, y))

        rooms = []
        
        # 2. Force the first room to be exactly where the player spawns
        px, py = PLAYER_START
        rooms.append((px - 1, py - 1, 3, 3)) # 3x3 starting room
        
        # 3. Generate a few random rooms
        num_rooms = random.randint(4, 6)
        for _ in range(num_rooms):
            w = random.randint(3, 6) # Width between 3 and 6 tiles
            h = random.randint(3, 6) # Height between 3 and 6 tiles
            x = random.randint(1, MAX_X - w)
            y = random.randint(1, MAX_Y - h)
            rooms.append((x, y, w, h))

        # 4. Carve out all the rooms by removing those walls
        for (rx, ry, rw, rh) in rooms:
            for x in range(rx, rx + rw):
                for y in range(ry, ry + rh):
                    walls.discard((x, y))
                    
        # 5. Connect the rooms with 1-tile wide L-shaped corridors
        for i in range(1, len(rooms)):
            prev_x, prev_y, pw, ph = rooms[i - 1]
            curr_x, curr_y, cw, ch = rooms[i]
            
            # Find the center of the previous room and the current room
            p_center_x, p_center_y = prev_x + pw // 2, prev_y + ph // 2
            c_center_x, c_center_y = curr_x + cw // 2, curr_y + ch // 2
            
            # Flip a coin to decide if we dig horizontal-then-vertical, or vice versa
            if random.choice([True, False]):
                # Horizontal dig
                for x in range(min(p_center_x, c_center_x), max(p_center_x, c_center_x) + 1):
                    walls.discard((x, p_center_y))
                # Vertical dig
                for y in range(min(p_center_y, c_center_y), max(p_center_y, c_center_y) + 1):
                    walls.discard((c_center_x, y))
            else:
                # Vertical dig
                for y in range(min(p_center_y, c_center_y), max(p_center_y, c_center_y) + 1):
                    walls.discard((p_center_x, y))
                # Horizontal dig
                for x in range(min(p_center_x, c_center_x), max(p_center_x, c_center_x) + 1):
                    walls.discard((x, c_center_y))
                
    return list(walls)

#
# CONSTS
#
PLAYER_CHAR = 'P'
EMPTY_CHAR = '_'
FOOD_CHAR = 'F'
WALL_CHAR = "X"
VOID_CHAR = ""

# Absolute Cardinal Directions
NORTH = (0, -1)
SOUTH = (0, 1)
EAST  = (1, 0)
WEST  = (-1, 0)

RELATIVE_DIRECTIONS = [
    "forward",
    "forward_left", 
    "left",
    "back_left",
    "back",
    "back_right",
    "right", 
    "forward_right"
]

NORTH_EAST = (1, -1)
NORTH_WEST = (-1, -1)
SOUTH_EAST = (1, 1)
SOUTH_WEST = (-1, 1)

# And add them to your FACING_NAMES dictionary
FACING_NAMES = {
    NORTH: "NORTH", SOUTH: "SOUTH", EAST: "EAST", WEST: "WEST",
    NORTH_EAST: "NORTH_EAST", NORTH_WEST: "NORTH_WEST", 
    SOUTH_EAST: "SOUTH_EAST", SOUTH_WEST: "SOUTH_WEST"
}

PLAYER_ARROWS = {
    NORTH: "▲",
    SOUTH: "▼",
    EAST:  "▶",
    WEST:  "◀"
}

#
# Classes/Harnesses
#

class World:
    """
    Represents the game world state used by a single simulation.
    The world stores the player, walls, and food in a sparse tile map,
    and provides movement plus perception utilities for the bug.
    """
    def __init__(self, 
                 max_x=MAX_X, 
                 max_y=MAX_Y, 
                 player_start=PLAYER_START, 
                 initial_food=None,
                 initial_walls=None,
                 facing=NORTH
                 ):
        self.state_dict = {}
        self.player_loc = player_start
        
        self.player_facing = facing 
        
        self.MAX_X = max_x
        self.MAX_Y = max_y

        self.state_dict[self.player_loc] = PLAYER_CHAR

        # --- Add the walls to the map ---
        if initial_walls:
            for wall in initial_walls:
                self.state_dict[wall] = WALL_CHAR

        if initial_food is None:
            initial_food = generate_initial_food(walls=initial_walls)

        for food in initial_food:
            self.state_dict[food] = FOOD_CHAR

    def draw_viewport(self, view_radius=5):
        """
        Draws an absolute, centered viewport around the player.
        This visualizes the portion of the world within the requested radius,
        marking walls, food, empty tiles, and the player's current facing.
        """
        (p_x, p_y) = self.player_loc
        facing_str = FACING_NAMES.get(self.player_facing, "UNKNOWN")
        
        print(f"--- Absolute World | Centered at ({p_x}, {p_y}) | Facing: {facing_str} ---")
        
        for y in range(p_y - view_radius, p_y + view_radius + 1):
            row_chars = []

            for x in range(p_x - view_radius, p_x + view_radius + 1):
                if x < 0 or x > self.MAX_X or y < 0 or y > self.MAX_Y:
                    row_chars.append(WALL_CHAR)
                elif (x, y) == self.player_loc:
                    row_chars.append(PLAYER_ARROWS.get(self.player_facing, PLAYER_CHAR))
                elif (x, y) in self.state_dict:
                    row_chars.append(self.state_dict[(x, y)])
                else:
                    row_chars.append(EMPTY_CHAR)
    
            print(" ".join(row_chars))

    def get_line_of_sight(self, distance, dx, dy):
        """
        Raycasts from the player's position in a single absolute direction.
        Returns the sequence of seen tiles up to the requested distance,
        stopping early at world boundaries, walls, or blocked diagonal corners.
        """
        p_x, p_y = self.player_loc
        results = []
    
        for step in range(1, distance + 1):
            target_x = p_x + (dx * step)
            target_y = p_y + (dy * step)
            
            # 1. Check World Bounds
            if target_x <= 0 or target_x >= MAX_X or target_y <= 0 or target_y >= MAX_Y:
                results.append(WALL_CHAR)
                break # Blocked by map edge
            
            # 2. Check for solid wall at destination
            target_tile = self.state_dict.get((target_x, target_y), EMPTY_CHAR)
            
            # --- Corner-Cutting Vision Check ---
            # If moving diagonally, check if we are "squeezing" through a blocked corner
            if abs(dx) == 1 and abs(dy) == 1:
                side_a = self.state_dict.get((p_x + (dx * step), p_y + (dy * (step - 1))), EMPTY_CHAR)
                side_b = self.state_dict.get((p_x + (dx * (step - 1)), p_y + (dy * step)), EMPTY_CHAR)
                
                # If both tiles beside the diagonal path are walls, the corner is solid
                if side_a == WALL_CHAR and side_b == WALL_CHAR:
                    results.append(WALL_CHAR)
                    break 

            # 3. Standard wall collision
            if target_tile == WALL_CHAR:
                results.append(WALL_CHAR)
                break # Blocked by wall
            else:
                results.append(target_tile)
                
        return results
    
    def get_perception(
        self, 
        forward=0, left=0, back=0, right=0, 
        forward_left=0, forward_right=0, back_left=0, back_right=0
    ):
        """
        Builds the bug's current sensory view in all eight relative directions.
        It converts the current facing into absolute ray vectors, then performs
        line-of-sight raycasts for each direction using the supplied vision distances.
        """
        fx, fy = self.player_facing
        
        # 1. Primary Vectors
        # Convert the current facing into absolute cardinal ray directions.
        forward_dx, forward_dy = fx, fy
        back_dx, back_dy       = -fx, -fy
        left_dx, left_dy       = fy, -fx
        right_dx, right_dy     = -fy, fx

        # 2. Diagonal Vectors 
        # --- Clamp the diagonals so the raycasts don't leap over tiles ---
        # Diagonal movement is represented by combining the cardinal vectors,
        # but then clamping to [-1, 1] so the raycast remains step-by-step.
        fl_dx = max(-1, min(1, forward_dx + left_dx))
        fl_dy = max(-1, min(1, forward_dy + left_dy))
        
        fr_dx = max(-1, min(1, forward_dx + right_dx))
        fr_dy = max(-1, min(1, forward_dy + right_dy))
        
        bl_dx = max(-1, min(1, back_dx + left_dx))
        bl_dy = max(-1, min(1, back_dy + left_dy))
        
        br_dx = max(-1, min(1, back_dx + right_dx))
        br_dy = max(-1, min(1, back_dy + right_dy))

        # 3. Raycast for all 8 directions
        return { 
            "forward":       self.get_line_of_sight(distance=forward,       dx=forward_dx, dy=forward_dy),
            "left":          self.get_line_of_sight(distance=left,          dx=left_dx,    dy=left_dy),
            "back":          self.get_line_of_sight(distance=back,          dx=back_dx,    dy=back_dy),
            "right":         self.get_line_of_sight(distance=right,         dx=right_dx,   dy=right_dy),
            "forward_left":  self.get_line_of_sight(distance=forward_left,  dx=fl_dx,      dy=fl_dy),
            "forward_right": self.get_line_of_sight(distance=forward_right, dx=fr_dx,      dy=fr_dy),
            "back_left":     self.get_line_of_sight(distance=back_left,     dx=bl_dx,      dy=bl_dy),
            "back_right":    self.get_line_of_sight(distance=back_right,    dx=br_dx,      dy=br_dy),
            "facing_absolute": FACING_NAMES.get(self.player_facing, "UNKNOWN")
        }
    
    def spawn_food(self):
        """
        Spawns a single piece of food in a random empty location inside the current world.
        The new food is only placed on tiles that are not currently occupied by the player,
        a wall, or existing food.
        """
        while True:
            # Pick a random spot inside the world boundaries
            new_x = random.randint(1, self.MAX_X - 1)
            new_y = random.randint(1, self.MAX_Y - 1)
            
            # If the spot is completely empty, place the food and exit loop
            if (new_x, new_y) not in self.state_dict:
                self.state_dict[(new_x, new_y)] = FOOD_CHAR
                # Log.info("New food spawned!", location=(new_x, new_y))
                break
    
    def move_relative(self, action):
        """
        Converts a relative movement command into an absolute move.
        This updates the bug's facing direction, enforces world boundaries and wall
        collisions, prevents illegal diagonal corner-cutting, and handles food consumption.
        """
        fx, fy = self.player_facing
        
        # 1a. Pre-calculate the 4 primary relative vectors
        f_dx, f_dy = fx, fy
        b_dx, b_dy = -fx, -fy
        l_dx, l_dy = fy, -fx
        r_dx, r_dy = -fy, fx

        # 1b. Calculate the absolute vector based on action
        if action == "forward":        dx, dy = f_dx, f_dy
        elif action == "back":         dx, dy = b_dx, b_dy
        elif action == "left":         dx, dy = l_dx, l_dy
        elif action == "right":        dx, dy = r_dx, r_dy
        elif action == "forward_left": dx, dy = f_dx + l_dx, f_dy + l_dy
        elif action == "forward_right":dx, dy = f_dx + r_dx, f_dy + r_dy
        elif action == "back_left":    dx, dy = b_dx + l_dx, b_dy + l_dy
        elif action == "back_right":   dx, dy = b_dx + r_dx, b_dy + r_dy
        else: return None

        # 1c. CLAMP THE VECTOR
        # This forces the movement to be at most 1 tile in any direction,
        # preventing "2 tile jumps" when turning from an already diagonal stance.
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))

        # 2. Calculate the target coordinates
        p_x, p_y = self.player_loc
        target_x = p_x + dx
        target_y = p_y + dy
        target_loc = (target_x, target_y)

        # 3. Handle Boundaries (Hitting a wall)
        if target_x <= 0 or target_x >= self.MAX_X or target_y <= 0 or target_y >= self.MAX_Y:
            self.player_facing = (dx, dy)
            return None

        # 4. Update orientation
        self.player_facing = (dx, dy)

        # 5. Check target square BEFORE moving
        target_content = self.state_dict.get(target_loc, EMPTY_CHAR)
        result = None
        
        # --- Stop the move if the destination itself is a wall ---
        if target_content == WALL_CHAR:
            return None 

        # --- Corner-Cutting Prevention ---
        # If moving diagonally, check the two adjacent cardinal tiles it is sliding past
        if abs(dx) == 1 and abs(dy) == 1:
            tile_x = self.state_dict.get((p_x + dx, p_y), EMPTY_CHAR)
            tile_y = self.state_dict.get((p_x, p_y + dy), EMPTY_CHAR)
            
            # If either adjacent tile is a wall, the bug cannot squeeze past the corner
            if tile_x == WALL_CHAR or tile_y == WALL_CHAR:
                return None
            
        # --- Check for food ---
        if target_content == FOOD_CHAR:
            result = "food"

            self.spawn_food()

        # 6. Execute the move
        if self.state_dict.get(self.player_loc) == PLAYER_CHAR:
            del self.state_dict[self.player_loc]
            
        self.player_loc = target_loc
        self.state_dict[self.player_loc] = PLAYER_CHAR
        
        return result
