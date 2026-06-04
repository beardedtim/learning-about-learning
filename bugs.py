from log import Log
import random
import json
import numpy as np
import concurrent.futures
import os

import torch
import torch.nn as nn
import copy
import torch.multiprocessing as tmp

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

# Preset vision configurations for different bug sensory styles.
# Each preset defines how far the bug can raycast in each relative direction.
# These values feed directly into World.get_perception() to build the bug's
# sensory input before it decides on an action.
VISION_CONES = {
    # THE BALANCED
    # Simulates a standard predator. Good forward visibility with just 
    # enough peripheral and diagonal vision to catch food as it walks past.
    "Balanced": {
        "forward": 5, 
        "left": 2, 
        "back": 1, 
        "right": 2, 
        "forward_left": 3,
        "forward_right": 3,
        "back_left": 1,
        "back_right": 1
    },

    # THE TUNNEL (SPRINTER)
    # Can see incredibly far ahead, but blind everywhere else. Diagonals 
    # are kept at 0 to strictly maintain the pure "laser beam" penalty. 
    # Great for bugs that move in fast, straight lines.
    "Tunnel": {
        "forward": 12, 
        "left": 0, 
        "back": 0, 
        "right": 0,
        "forward_left": 0,
        "forward_right": 0,
        "back_left": 0,
        "back_right": 0
    },

    # THE PREY (PERIPHERAL)
    # Mimics animals like horses or rabbits with eyes on the sides of their heads. 
    # Terrible depth perception directly in front, but massive side-to-side awareness.
    # Strong diagonals create a wide "hammerhead shark" field of view.
    "Prey": {
        "forward": 2, 
        "left": 5, 
        "back": 2, 
        "right": 5,
        "forward_left": 4,
        "forward_right": 4,
        "back_left": 4,
        "back_right": 4
    },

    # THE RADAR (OMNISCIENT)
    # Perfect 360-degree awareness, but sacrifices long-range sight. The diagonal 
    # values match the cardinals perfectly to create a consistent, omniscient aura.
    "Radar": {
        "forward": 4, 
        "left": 4, 
        "back": 4, 
        "right": 4,
        "forward_left": 4,
        "forward_right": 4,
        "back_left": 4,
        "back_right": 4
    }
}

# The default vision cone used when a bug does not specify one explicitly.
# This is a shallow reference to the Balanced preset above.
DEFAULT_VISION_CONE = VISION_CONES.get("Balanced")


def generate_initial_food(walls=None, num_items=DEFAULT_INITIAL_FOOD_COUNT):
    """
    Randomly generates a list of unique food coordinates.
    Ensures food does not spawn on the player or inside a wall.
    """
    food_positions = set()
    wall_set = set(walls) if walls else set()
    
    while len(food_positions) < num_items:
        x = random.randint(1, MAX_X - 1)
        y = random.randint(1, MAX_Y - 1)
        new_pos = (x, y)
        
        # --- Check that the spot isn't a wall ---
        if new_pos != PLAYER_START and new_pos not in food_positions and new_pos not in wall_set:
            food_positions.add(new_pos)
            
    return list(food_positions)

def generate_walls(layout="empty"):
    """Generates a list of coordinates for wall placements based on a layout preset."""
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

# How many turns a bug can survive without eating before it starves.
# This counter resets to the maximum whenever the bug consumes food.
LIFE_FORCE = 25

# Hard deadline for a single simulation run. Prevents infinite wandering
# and keeps fitness evaluation bounded in time.
MAX_ITERATIONS = 10000

#
# Genetic algorithm default hyperparameters
#
# GENERATIONS: number of evolutionary rounds to run.
# POP_SIZE: number of candidates evaluated each generation.
# MUTATION_RATE: strength of random variation applied during breeding.
# TRIALS_PER_EPOCH: number of separate random worlds each bug plays per fitness estimate.
GENERATIONS = 100
POP_SIZE = 1250
MUTATION_RATE = 0.075
TRIALS_PER_EPOCH = 5

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

FACING_NAMES = {
    NORTH: "NORTH",
    SOUTH: "SOUTH",
    EAST:  "EAST",
    WEST:  "WEST"
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
            # Track the score and spawn a new piece of food
            if not hasattr(self, 'food_collected'): 
                self.food_collected = 0
            self.food_collected += 1
            self.spawn_food()

        # 6. Execute the move
        if self.state_dict.get(self.player_loc) == PLAYER_CHAR:
            del self.state_dict[self.player_loc]
            
        self.player_loc = target_loc
        self.state_dict[self.player_loc] = PLAYER_CHAR
        
        return result

class BaseBug:
    def __init__(self, vision_cone=DEFAULT_VISION_CONE):
        self.vision_cone = vision_cone

    def print_bug_perspective(self, perspective):
        """
        Draws a crosshair representing what the bug can currently see relative to its facing.
        """
        # Exclude the string key when calculating grid size
        sightlines = [v for k, v in perspective.items() if isinstance(v, list)]
        max_dist = max(len(line) for line in sightlines) if sightlines else 0
        
        grid_size = (max_dist * 2) + 1
        grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
        
        center = max_dist
        
        # In a relative HUD, "forward" is ALWAYS up. So we always use the UP arrow here.
        grid[center][center] = "▲" 
        
        for i, tile in enumerate(perspective.get("forward", [])):
            grid[center - 1 - i][center] = tile
            
        for i, tile in enumerate(perspective.get("back", [])):
            grid[center + 1 + i][center] = tile
            
        for i, tile in enumerate(perspective.get("left", [])):
            grid[center][center - 1 - i] = tile
            
        for i, tile in enumerate(perspective.get("right", [])):
            grid[center][center + 1 + i] = tile
            
        facing_str = perspective.get("facing_absolute", "UNKNOWN")
        print(f"--- Bug's HUD (Relative) | Compass: {facing_str} ---")
        
        for row in grid:
            if any(char != " " for char in row):
                print(" ".join(row))
        print("-----------------\n")

class RandomBug(BaseBug):
    def __init__(self, vision_cone=DEFAULT_VISION_CONE):
        super().__init__(vision_cone)

    def request_action(self, perception):
        scan_order = [
            "forward", "forward_left", "forward_right", 
            "left", "right", 
            "back_left", "back_right", "back"
        ]
        
        for direction in scan_order:
            if FOOD_CHAR in perception.get(direction, []):
                return direction
        
        return random.choice(RELATIVE_DIRECTIONS)


class ForwardBug(BaseBug):
    def __init__(self, vision_cone=DEFAULT_VISION_CONE):
        super().__init__(vision_cone)

    def request_action(self, perception):
        # INSTINCT 1: Survival 
        # Scan all 8 directions, heavily prioritizing forward momentum
        scan_order = [
            "forward", "forward_left", "forward_right", 
            "left", "right", 
            "back_left", "back_right", "back"
        ]
        
        for direction in scan_order:
            if FOOD_CHAR in perception.get(direction, []):
                return direction
        
        # INSTINCT 2: Momentum
        # Look at the immediate next square (index 0) in the forward sightline
        forward_view = perception.get("forward", [])
        
        # If there is no wall immediately in front of us...
        if len(forward_view) > 0 and forward_view[0] != WALL_CHAR:
            
            # 85% chance to just keep going straight. 
            if random.random() < 0.85:
                return "forward"
        
        # INSTINCT 3: Obstacle Avoidance / Organic Turning
        # If we reached here, forward is blocked by a wall, OR we rolled the 15% chance to turn.
        safe_moves = []
        
        turn_options = ["forward_left", "forward_right", "left", "right"]
        
        for direction in turn_options:
            view = perception.get(direction, [])
            # A move is safe if we have vision there AND the immediate next step isn't a wall
            if len(view) > 0 and view[0] != WALL_CHAR:
                safe_moves.append(direction)
        
        # Pick a random safe direction
        if safe_moves:
            return random.choice(safe_moves)
        
        # INSTINCT 4: Trapped!
        # If forward, left, right, and forward-diagonals are all walls, it's a dead end.
        return "back"

class BrainBug(BaseBug):
    """
    A Brain Bug has _simple_ input -> action brain. It can be modified
    with Rules based on those inputs but isn't allowed to do get much
    more complex than that
    """
    def __init__(self, vision_cone, genes=None):
        super().__init__(vision_cone)
        
        # If no genes are provided, spawn with a completely random brain
        if genes is None:
            self.genes = {
                "food_weight": random.uniform(0.1, 1.0),
                "wall_weight": random.uniform(-1.0, 1.0),
                "empty_weight": random.uniform(-1.0, 1.0)
            }
        else:
            self.genes = genes

    def request_action(self, perception):
        best_direction = "forward"
        highest_score = -9999
        
        directions = [
            "forward", "forward_left", "forward_right", 
            "left", "right", 
            "back_left", "back_right", "back"
        ]
        
        for direction in directions:
            score = 0
            view = perception.get(direction, [])
            found_food = False
            
            # Scan EVERY tile in our line of sight for this direction
            for distance_index, tile in enumerate(view):
                # Distance is index + 1 (so the immediate tile is distance 1)
                distance = distance_index + 1 
                
                if tile == FOOD_CHAR:
                    # Closer food gives a higher score!
                    score += self.genes["food_weight"] / distance
                    found_food = True
                    break # We found food, stop scanning this direction
                    
                elif tile == WALL_CHAR:
                    # Penalize walls blocking this direction
                    score += self.genes["wall_weight"]
                    break # You can't see past a wall, stop scanning
            
            # If we looked down this path and saw no food, apply the empty space weight
            if not found_food:
                score += self.genes["empty_weight"]
            
            # Update our best choice
            if score > highest_score:
                highest_score = score
                best_direction = direction
                
        return best_direction

    def mutate(self, mutation_rate=0.1):
        """Creates a slightly mutated copy of this bug's genes for its offspring."""
        new_genes = {}
        for gene_name, weight in self.genes.items():
            # Add a small random tweak to the gene (e.g., +/- 10%)
            tweak = random.uniform(-mutation_rate, mutation_rate)
            new_genes[gene_name] = weight + tweak
            
        return new_genes
    
    def spawn_child(self, mutation_rate, other_parent=None):
        """Returns a brand new BrainBug with mixed and mutated genes."""
        
        # 1. Mix the genes
        mixed_genes = {}
        for key in self.genes:
            if other_parent is not None and random.random() > 0.5:
                mixed_genes[key] = other_parent.genes[key]
            else:
                mixed_genes[key] = self.genes[key]
                
        # 2. Mutate the mixed genes
        mutated_genes = {}
        for gene_name, weight in mixed_genes.items():
            tweak = random.uniform(-mutation_rate, mutation_rate)
            mutated_genes[gene_name] = weight + tweak
            
        return BrainBug(vision_cone=self.vision_cone, genes=mutated_genes)
    
    def save_to_file(self, filename):
        """Saves the bug's vision cone and genetic weights to a JSON file."""
        data = {
            "bug_type": "BrainBug",
            "vision_cone": self.vision_cone,
            "genes": self.genes
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        Log.info(f"BrainBug saved successfully to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        """Loads a JSON file and returns a fully reconstructed BrainBug."""
        with open(filename, 'r') as f:
            data = json.load(f)
            
        return cls(
            vision_cone=data["vision_cone"], 
            genes=data["genes"]
        )

class NumpyNeuralNet:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # If no weights are provided, generate a completely random brain using NumPy
        if weights is None:
            # np.random.uniform generates the entire matrix instantly
            self.W1 = np.random.uniform(-1, 1, (hidden_size, input_size)).astype(np.float32)
            self.b1 = np.random.uniform(-1, 1, hidden_size).astype(np.float32)
            self.W2 = np.random.uniform(-1, 1, (output_size, hidden_size)).astype(np.float32)
            self.b2 = np.random.uniform(-1, 1, output_size).astype(np.float32)
        else:
            self.W1, self.b1, self.W2, self.b2 = weights

    def forward(self, inputs):
        """Passes the vision data through the network using Vectorized math."""
        # 1. Hidden Layer Math: Z = W * X + b
        z1 = np.dot(self.W1, inputs) + self.b1
        
        # ReLU Activation Function (converts negative numbers to 0)
        a1 = np.maximum(0, z1)

        # 2. Output Layer Math
        z2 = np.dot(self.W2, a1) + self.b2

        # --- THE FIX: THE SQUISH ---
        # Apply Tanh to prevent the memory outputs from exploding to Infinity!
        a2 = np.tanh(z2)

        # Convert back to a standard Python list
        return a2.tolist()

    def mutate(self, rate=0.05):
        """Creates a slightly altered copy of this brain for offspring."""
        # NumPy allows us to add a matrix of random noise to our entire weight matrix in one line
        new_W1 = self.W1 + np.random.uniform(-rate, rate, self.W1.shape)
        new_b1 = self.b1 + np.random.uniform(-rate, rate, self.b1.shape)
        new_W2 = self.W2 + np.random.uniform(-rate, rate, self.W2.shape)
        new_b2 = self.b2 + np.random.uniform(-rate, rate, self.b2.shape)
        
        return NumpyNeuralNet(self.input_size, self.hidden_size, self.output_size, (new_W1, new_b1, new_W2, new_b2))
    
    def crossover(self, other_parent_brain):
        """Merges this brain with another by flipping a coin for each NEURON (Row-wise)."""
        
        # 1. Flip a coin for each Hidden Neuron
        # We create a mask of shape (hidden_size, 1) so it broadcasts across the entire row
        mask_hidden = np.random.rand(self.hidden_size, 1) > 0.5
        
        # Weave W1 (The entire row is taken from self or other_parent)
        child_W1 = np.where(mask_hidden, self.W1, other_parent_brain.W1)
        # Flatten the mask for the 1D bias array
        child_b1 = np.where(mask_hidden.flatten(), self.b1, other_parent_brain.b1)
        
        # 2. Flip a coin for each Output Neuron
        mask_output = np.random.rand(self.output_size, 1) > 0.5
        
        # Weave W2
        child_W2 = np.where(mask_output, self.W2, other_parent_brain.W2)
        child_b2 = np.where(mask_output.flatten(), self.b2, other_parent_brain.b2)

        # 3. Return a completely new brain
        return NumpyNeuralNet(
            self.input_size, self.hidden_size, self.output_size, 
            (child_W1, child_b1, child_W2, child_b2)
        )

class NeuralBug(BaseBug):
    """
    A perceptron-based bug with a fixed-size neural network brain.

    NeuralBug converts relative perception into a 17-element input vector:
    - 8 food proximity values
    - 8 immediate wall indicators
    - 1 normalized life force value

    The brain emits 8 output scores, one per relative movement direction.
    The chosen action is the direction with the highest output.
    """
    def __init__(self, vision_cone, brain=None):
        super().__init__(vision_cone)
        
        # The direction order is critical: it must match the brain's input/output mapping.
        self.directions = [
            "forward", "forward_left", "forward_right", 
            "left", "right", 
            "back_left", "back_right", "back"
        ]
        
        # 17 Inputs (8 food, 8 walls, 1 life force), 12 Hidden Neurons, 8 Outputs (movement choices)
        if brain is None:
            self.brain = NumpyNeuralNet(input_size=17, hidden_size=12, output_size=8)
        else:
            self.brain = brain

    def request_action(self, perception):
        """
        Converts the current perception into a neural input vector, evaluates the
        brain, and returns the relative movement direction with the highest score.
        """
        inputs = []
        
        # --- BUILD THE INPUT VECTOR (17 numbers) ---
        # 8 food signals followed by 8 wall signals and 1 life force value.
        food_inputs = []
        wall_inputs = []
        
        for direction in self.directions:
            view = perception.get(direction, [])
            food_score = 0.0
            wall_score = 0.0
            
            for distance_index, tile in enumerate(view):
                distance = distance_index + 1 
                
                if tile == FOOD_CHAR and food_score == 0:
                    food_score = 1.0 / distance # Closer food = higher score
                
                if tile == WALL_CHAR:
                    if distance == 1:
                        wall_score = 1.0 # 1.0 means a wall is touching us
                    break # Stop scanning - you can't see past walls
                    
            food_inputs.append(food_score)
            wall_inputs.append(wall_score)
        # If the simulation hasn't started yet, default to 1.0 (full)
        current_life = getattr(self, 'life_force', 100)
        max_life = getattr(self, 'max_life_force', 100)
        normalized_hunger = current_life / max_life

        # Combine them into a single list of 17 numbers (8 food + 8 walls + 1 life force).
        inputs = np.array(food_inputs + wall_inputs + [normalized_hunger], dtype=np.float32)

        # --- THINK ---
        # Pass the inputs through the brain to get 8 output scores.
        output_scores = self.brain.forward(inputs)

        # --- ACT ---
        # Choose the movement with the highest output score.
        best_index = output_scores.index(max(output_scores))

        # --- STORE ---
        self.last_action_scores = output_scores
        self.last_perception = perception

        return self.directions[best_index]

    def mutate(self, mutation_rate=0.05):
        """
        Returns a mutated copy of this bug's neural brain.
        This helper is used by reproduction routines to create variation.
        """
        mutated_brain = self.brain.mutate(rate=mutation_rate)

        return mutated_brain
    
    def spawn_child(self, mutation_rate, other_parent=None):
        """
        Produces a new NeuralBug offspring from this bug's brain.
        If another parent is provided, the offspring uses crossover to mix brains.
        Otherwise it clones this bug's brain and applies mutation.
        """
        
        # 1. If we have a mate, mix the brains together
        if other_parent is not None:
            mixed_brain = self.brain.crossover(other_parent.brain)
        else:
            # Asexual fallback (just copy our own brain)
            mixed_brain = NumpyNeuralNet(
                self.brain.input_size, self.brain.hidden_size, self.brain.output_size,
                (self.brain.W1.copy(), self.brain.b1.copy(), self.brain.W2.copy(), self.brain.b2.copy())
            )
            
        # 2. Mutate the resulting brain
        mutated_brain = mixed_brain.mutate(mutation_rate)
        
        return NeuralBug(vision_cone=self.vision_cone, brain=mutated_brain)
    
    def save_to_file(self, filename):
        """Saves the bug's vision cone and neural network weights to a JSON file."""
        data = {
            "bug_type": "NeuralBug",
            "vision_cone": self.vision_cone,
            "brain": {
                "input_size": self.brain.input_size,
                "hidden_size": self.brain.hidden_size,
                "output_size": self.brain.output_size,
                # --- Convert NumPy arrays back to standard Python lists for JSON ---
                "weights": [
                    self.brain.W1.tolist(), 
                    self.brain.b1.tolist(), 
                    self.brain.W2.tolist(), 
                    self.brain.b2.tolist()
                ]
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        Log.info(f"NeuralBug saved successfully to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        """Loads a JSON file and returns a fully reconstructed NeuralBug."""
        with open(filename, 'r') as f:
            data = json.load(f)
            
        brain_data = data["brain"]
        saved_weights = brain_data["weights"]
        
        # --- Convert the loaded JSON lists back into fast NumPy arrays ---
        numpy_weights = (
            np.array(saved_weights[0], dtype=np.float32),
            np.array(saved_weights[1], dtype=np.float32),
            np.array(saved_weights[2], dtype=np.float32),
            np.array(saved_weights[3], dtype=np.float32),
        )
        
        reconstructed_brain = NumpyNeuralNet(
            input_size=brain_data["input_size"],
            hidden_size=brain_data["hidden_size"],
            output_size=brain_data["output_size"],
            weights=numpy_weights
        )
        
        return cls(
            vision_cone=data["vision_cone"], 
            brain=reconstructed_brain
        )


class MemoryBug(BaseBug):
    """
    A NeuralBug extension with a recurrent memory vector.

    MemoryBug appends a small scratchpad of previous outputs to its input vector.
    This allows it to carry short-term state across turns without a more complex
    recurrent network architecture.

    Input vector layout:
    - 8 food proximity values
    - 8 wall indicators
    - 1 normalized life force value
    - N memory values (default 4)

    Output vector layout:
    - 8 movement scores
    - N memory values to retain for the next turn
    """
    def __init__(self, vision_cone, brain=None, memory_size=4):
        super().__init__(vision_cone)
        
        self.directions = [
            "forward", "forward_left", "forward_right", 
            "left", "right", 
            "back_left", "back_right", "back"
        ]
        
        self.memory_size = memory_size
        
        # The bug starts with a blank memory (all zeros)
        self.memory = np.zeros(self.memory_size)
        
        if brain is None:
            # 16 vision + 1 life force + 4 memory = 21 Inputs
            # 8 movement outputs + 4 memory outputs = 12 Outputs
            self.brain = NumpyNeuralNet(
                input_size=17 + self.memory_size, 
                hidden_size=20, 
                output_size=8 + self.memory_size
            )
        else:
            self.brain = brain

    def reset_memory(self):
        """
        Clears the recurrent memory buffer before a new simulation starts.
        This ensures each run begins with a blank internal state.
        """
        self.memory = np.zeros(self.memory_size)

    def request_action(self, perception):
        """
        Builds the extended input vector, runs the brain, and returns the
        relative movement direction with the highest score.

        The MemoryBug reuses its previous memory outputs as additional inputs
        on the next turn, enabling short-term continuity across decisions.
        """
        # 1. Build the vision inputs (16 numbers) just like before
        food_inputs = []
        wall_inputs = []
        
        for direction in self.directions:
            view = perception.get(direction, [])
            food_score = 0.0
            wall_score = 0.0
            
            for distance_index, tile in enumerate(view):
                distance = distance_index + 1 
                
                if tile == FOOD_CHAR and food_score == 0:
                    food_score = 1.0 / distance 
                
                if tile == WALL_CHAR:
                    if distance == 1:
                        wall_score = 1.0 
                    break # Stop scanning - you can't see past walls
                    
            food_inputs.append(food_score)
            wall_inputs.append(wall_score)
            
        # 2. Combine vision inputs and current life force into 17 values
        current_life = getattr(self, 'life_force', 100)
        max_life = getattr(self, 'max_life_force', 100)
        normalized_hunger = current_life / max_life
        
        vision_inputs = food_inputs + wall_inputs + [normalized_hunger]

        # Combine Vision (17) + Memory (4) = 21 numbers total
        full_inputs = vision_inputs + self.memory.tolist()

        # --- THINK ---
        outputs = self.brain.forward(full_inputs)

        # 4. SPLIT THE OUTPUTS
        # The first 8 numbers are our movement choices
        action_scores = outputs[:8]
        
        # The remaining numbers are our new memories to save for next turn
        new_memory = outputs[8:]
        self.memory = np.array(new_memory)

        # 5. ACT
        best_index = action_scores.index(max(action_scores))

        # 6: STORE
        self.last_vision = vision_inputs
        self.last_action_scores = action_scores
        self.last_perception = perception
        
        return self.directions[best_index]

    def spawn_child(self, mutation_rate, other_parent=None):
        """
        Produces a new MemoryBug offspring from this bug's brain.
        If another parent is provided, the offspring uses crossover to mix brains.
        Otherwise it clones this bug's brain and applies mutation.
        The child's memory_size is preserved from the parent.
        """
        if other_parent is not None:
            mixed_brain = self.brain.crossover(other_parent.brain)
        else:
            mixed_brain = NumpyNeuralNet(
                self.brain.input_size, self.brain.hidden_size, self.brain.output_size,
                (self.brain.W1.copy(), self.brain.b1.copy(), self.brain.W2.copy(), self.brain.b2.copy())
            )
            
        mutated_brain = mixed_brain.mutate(mutation_rate)
        
        return MemoryBug(
            vision_cone=self.vision_cone, 
            brain=mutated_brain, 
            memory_size=self.memory_size
        )
    
    def save_to_file(self, filename):
        """Saves the bug's vision cone, memory size, and weights to a JSON file."""
        data = {
            "bug_type": "MemoryBug",
            "vision_cone": self.vision_cone,
            "memory_size": self.memory_size,
            "brain": {
                "input_size": self.brain.input_size,
                "hidden_size": self.brain.hidden_size,
                "output_size": self.brain.output_size,
                "weights": [
                    self.brain.W1.tolist(), 
                    self.brain.b1.tolist(), 
                    self.brain.W2.tolist(), 
                    self.brain.b2.tolist()
                ]
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        Log.info(f"MemoryBug saved successfully to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        """Loads a JSON file and returns a fully reconstructed MemoryBug."""
        with open(filename, 'r') as f:
            data = json.load(f)
            
        brain_data = data["brain"]
        saved_weights = brain_data["weights"]
        
        numpy_weights = (
            np.array(saved_weights[0], dtype=np.float32),
            np.array(saved_weights[1], dtype=np.float32),
            np.array(saved_weights[2], dtype=np.float32),
            np.array(saved_weights[3], dtype=np.float32),
        )
        
        reconstructed_brain = NumpyNeuralNet(
            input_size=brain_data["input_size"],
            hidden_size=brain_data["hidden_size"],
            output_size=brain_data["output_size"],
            weights=numpy_weights
        )
        
        return cls(
            vision_cone=data["vision_cone"], 
            brain=reconstructed_brain,
            memory_size=data.get("memory_size", 4)
        )


# ───────────────────────────────────────────────────────────────────────────────
# TorchBrain
# ───────────────────────────────────────────────────────────────────────────────
 
class TorchBrain(nn.Module):
    """
    A multi-layer GRU brain for TorchBug.
 
    Architecture
    ────────────
    Input  → GRU (N layers, hidden_size units) → Linear head → 8 action logits
 
    The GRU hidden state carries memory across turns automatically.
    No manual memory feedback vector needed — that's the whole point.
 
    Parameters
    ──────────
    input_size  : number of floats in the per-turn perception vector (default 17)
    hidden_size : width of each GRU layer
    num_layers  : depth of the GRU stack (2 is a good starting point)
    output_size : number of movement choices (always 8 to match RELATIVE_DIRECTIONS)
    """
    def __init__(self, input_size=17, hidden_size=32, num_layers=2, output_size=8):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_size = output_size
 
        self.gru  = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)
    
    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__init__(
            input_size  = state['input_size'],
            hidden_size = state['hidden_size'],
            num_layers  = state['num_layers'],
            output_size = state['output_size'],
        )
        sd = {k: torch.tensor(v) for k, v in state['state_dict'].items()}
        self.load_state_dict(sd)

    def forward(self, x, hidden=None):
        """
        x      : (input_size,) float tensor — one turn's perception
        hidden : GRU hidden state from the previous turn, or None to start fresh
 
        Returns
        ───────
        logits : (output_size,) float tensor — raw action scores
        hidden : updated hidden state to pass in next turn
        """
        # GRU expects (batch, seq, features) — we treat one turn as a sequence of 1
        out, hidden = self.gru(x.view(1, 1, -1), hidden)
        logits = self.head(out.squeeze())   # (output_size,)
        return logits, hidden
 
    # ── Evolutionary operators ─────────────────────────────────────────────────
 
    def clone(self):
        """Returns a deep copy of this brain with identical weights."""
        return copy.deepcopy(self)
 
    def mutate(self, rate=0.05):
        """
        Returns a new TorchBrain with Gaussian noise added to every parameter.
        'rate' is the std-dev of the noise — equivalent to mutation_rate elsewhere.
        """
        child = self.clone()
        with torch.no_grad():
            for param in child.parameters():
                param.add_(torch.randn_like(param) * rate)
        return child
 
    def crossover(self, other):
        """
        Returns a new TorchBrain by flipping a coin for each *parameter tensor*
        (not each weight individually, which keeps layer structure coherent).
        Then mutates the result slightly.
        """
        child = self.clone()
        with torch.no_grad():
            for p_child, p_other in zip(child.parameters(), other.parameters()):
                if random.random() > 0.5:
                    p_child.copy_(p_other)
        return child
 
    # ── Persistence ───────────────────────────────────────────────────────────
 
    def to_dict(self):
        """Serialises weights to plain Python lists so they can be JSON-dumped."""
        return {
            "input_size":  self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "output_size": self.output_size,
            "state_dict":  {k: v.tolist() for k, v in self.state_dict().items()}
        }
 
    @classmethod
    def from_dict(cls, d):
        """Reconstructs a TorchBrain from the dict produced by to_dict()."""
        brain = cls(
            input_size  = d["input_size"],
            hidden_size = d["hidden_size"],
            num_layers  = d["num_layers"],
            output_size = d["output_size"],
        )
        state = {k: torch.tensor(v) for k, v in d["state_dict"].items()}
        brain.load_state_dict(state)
        return brain
 

class TorchBug(BaseBug):
    """
    A bug driven by a multi-layer GRU (TorchBrain).
 
    Input vector layout (17 floats, same as NeuralBug):
    ────────────────────────────────────────────────────
      [0:8]  food proximity scores  (1/distance, 0 if none seen)
      [8:16] wall indicators        (1.0 if wall is touching, else 0)
      [16]   normalised life force  (life_force / max_life_force)
 
    The GRU hidden state replaces the hand-rolled memory vector from MemoryBug.
    Hidden state is reset at the start of each simulation via reset_memory().
 
    Hyperparameters
    ───────────────
    hidden_size : width of GRU layers (default 32)
    num_layers  : depth of GRU stack  (default 2)
    """
 
    INPUT_SIZE  = 17
    OUTPUT_SIZE = 8   # one score per relative direction
    def __getstate__(self):
        state = self.__dict__.copy()
        # Serialize brain to plain dicts/lists instead of live tensors
        state['brain'] = self.brain.to_dict()
        state['_hidden'] = None  # never pickle hidden state
        return state
    
    def __setstate__(self, state):
        state['brain'] = TorchBrain.from_dict(state['brain'])
        self.__dict__.update(state)

    def __init__(self, vision_cone=DEFAULT_VISION_CONE, brain=None,
                 hidden_size=32, num_layers=2):
        super().__init__(vision_cone)
 
        self.directions = [
            "forward", "forward_left", "forward_right",
            "left", "right",
            "back_left", "back_right", "back"
        ]
 
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
 
        if brain is None:
            self.brain = TorchBrain(
                input_size  = self.INPUT_SIZE,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                output_size = self.OUTPUT_SIZE,
            )
        else:
            self.brain = brain
 
        # Hidden state — reset between simulations
        self._hidden = None
 
    # ── Simulation interface ───────────────────────────────────────────────────
 
    def reset_memory(self):
        """
        Called by Simulation.run() before each trial.
        Wipes the GRU hidden state so every run starts from a blank slate.
        """
        self._hidden = None
 
    def _build_input(self, perception):
        """Converts a perception dict into a (17,) float32 tensor."""
        food_inputs = []
        wall_inputs = []
 
        for direction in self.directions:
            view       = perception.get(direction, [])
            food_score = 0.0
            wall_score = 0.0
 
            for distance_index, tile in enumerate(view):
                distance = distance_index + 1
 
                if tile == FOOD_CHAR and food_score == 0:
                    food_score = 1.0 / distance
 
                if tile == WALL_CHAR:
                    if distance == 1:
                        wall_score = 1.0
                    break   # can't see past a wall
 
            food_inputs.append(food_score)
            wall_inputs.append(wall_score)
 
        current_life       = getattr(self, 'life_force', 100)
        max_life           = getattr(self, 'max_life_force', 100)
        normalised_hunger  = current_life / max_life
 
        raw = food_inputs + wall_inputs + [normalised_hunger]
        return torch.tensor(raw, dtype=torch.float32)
 
    def request_action(self, perception):
        """
        Builds the input vector, steps the GRU forward by one turn,
        and returns the direction with the highest output score.
        """
        x = self._build_input(perception)
 
        with torch.no_grad():
            logits, self._hidden = self.brain(x, self._hidden)
 
        best_index = logits.argmax().item()
 
        # Store for debugging / visualisation
        self.last_action_scores = logits.tolist()
        self.last_perception    = perception
 
        return self.directions[best_index]
 
    # ── Evolutionary operators ─────────────────────────────────────────────────
 
    def mutate(self, mutation_rate=0.05):
        """Returns a new TorchBug with a mutated copy of this brain."""
        mutated_brain = self.brain.mutate(rate=mutation_rate)
        return TorchBug(
            vision_cone = self.vision_cone,
            brain       = mutated_brain,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
        )
 
    def spawn_child(self, mutation_rate, other_parent=None):
        """
        Produces an offspring TorchBug.
        With a second parent: crossover then mutate.
        Without: clone then mutate (asexual fallback).
        """
        if other_parent is not None:
            mixed_brain = self.brain.crossover(other_parent.brain)
        else:
            mixed_brain = self.brain.clone()
 
        mutated_brain = mixed_brain.mutate(rate=mutation_rate)
 
        return TorchBug(
            vision_cone = self.vision_cone,
            brain       = mutated_brain,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
        )
 
    # ── Persistence ───────────────────────────────────────────────────────────
 
    def save_to_file(self, filename):
        """Saves the bug's vision cone and brain weights to a JSON file."""
        data = {
            "bug_type":    "TorchBug",
            "vision_cone": self.vision_cone,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "brain":       self.brain.to_dict(),
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
 
    @classmethod
    def load_from_file(cls, filename):
        """Loads a JSON file and returns a fully reconstructed TorchBug."""
        with open(filename, 'r') as f:
            data = json.load(f)
 
        brain = TorchBrain.from_dict(data["brain"])
 
        return cls(
            vision_cone = data["vision_cone"],
            brain       = brain,
            hidden_size = data["hidden_size"],
            num_layers  = data["num_layers"],
        )
 


class Simulation:
    def __init__(self, world, bug, max_iterations=MAX_ITERATIONS, life_force=LIFE_FORCE):
        self.world = world
        self.bug = bug
        self.max_iterations = max_iterations
        self.life_force = life_force
        
    def run(self):
        """Runs the bug through the world until it starves or hits the iteration limit."""
        self.bug.max_life_force = self.life_force
        self.bug.life_force = self.life_force

        # Reset the bug's memory before we run this simulation
        if hasattr(self.bug, 'reset_memory'):
            self.bug.reset_memory()
        
        turns_survived = 0

        for turn in range(self.max_iterations):
            self.bug.life_force -= 1
            turns_survived += 1
            
            # 1. Look
            perception = self.world.get_perception(**self.bug.vision_cone)
            
            # 2. Think
            next_action = self.bug.request_action(perception=perception)
            
            # 3. Act
            move_result = self.world.move_relative(next_action)
            
            # 4. React (Restore the bug's life force if it eats)
            if move_result == "food":
                self.bug.life_force = self.bug.max_life_force
        
            # 5. Check Survival
            if self.bug.life_force <= 0:
                break
                
        # Return standard metrics so the trainer knows what happened
        food_collected = getattr(self.world, 'food_collected', 0)
        
        return {
            "turns_survived": turns_survived,
            "food_collected": food_collected,
            "starved": self.bug.life_force <= 0
        }

#
# Brokwn out so that we can run in multiprocessing
#
def evaluate_single_bug_worker(args):
    # --- Accept map_layout ---
    tmp.set_sharing_strategy('file_system')
    bug, trials, fitness_fn, map_layout = args
    total_fitness = 0
    
    for _ in range(trials):
        # --- Generate the map state sequentially ---
        walls = generate_walls(map_layout)
        food = generate_initial_food(walls=walls)
        world = World(initial_food=food, initial_walls=walls)
        
        sim = Simulation(world, bug)
        results = sim.run()
        
        trial_score = fitness_fn(world, results["turns_survived"])
        total_fitness += trial_score
        
    return total_fitness / trials

class EvolutionaryTrainer:
    def __init__(self, 
                 bug_class, 
                 vision_cone, 
                 fitness_fn, 
                 generations=GENERATIONS, 
                 population_size=POP_SIZE, 
                 mutation_rate=MUTATION_RATE, 
                 trials=1,
                 map_layout="empty",
                 name=None):
        
        self.bug_class = bug_class
        self.vision_cone = vision_cone
        self.fitness_fn = fitness_fn
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.trials = trials
        self.map_layout = map_layout
        
        self.overall_best_score = -9999999 
        self.name = name

    def train(self):
        fitness_name = self.fitness_fn.__name__.upper()
        bug_name = self.bug_class.__name__
        
        Log.info(f"--- STARTING {bug_name} EVOLUTION ({fitness_name}) {self.name} ---")
        Log.info(f"Generations: {self.generations} | Population: {self.population_size} | Trials: {self.trials}")

        # 1. Initialize Generation 0 using the provided class
        population = [self.bug_class(vision_cone=self.vision_cone) for _ in range(self.population_size)]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for gen in range(self.generations):
                # 2. Evaluate Fitness via Trials
                
                # Pack up the arguments each CPU core needs into a tuple
                worker_args = [
                    (bug, self.trials, self.fitness_fn, self.map_layout)
                    for bug in population
                ]
                
                chunksize = max(1, self.population_size // (os.cpu_count() * 4))
                fitness_scores = list(executor.map(evaluate_single_bug_worker, worker_args, chunksize=chunksize))
                    
                # Assign the calculated scores back to our main population
                for bug, score in zip(population, fitness_scores):
                    bug.fitness = score

                # 3. Sort Population (Highest fitness first)
                population.sort(key=lambda b: getattr(b, 'fitness', -999999), reverse=True)
                
                top_score = population[0].fitness
                avg_score = sum(b.fitness for b in population) / self.population_size
                
                Log.info("Generation stats", 
                    gen=gen + 1, 
                    top_score=f"{top_score:.1f}", 
                    avg_score=f"{avg_score:.2f}")
                
                if top_score > self.overall_best_score:
                    self.overall_best_score = top_score
                    Log.info(f"New Apex Bug! ({fitness_name} Score: {top_score:.1f})")

                # 4. Selection and Mutation
                num_parents = max(2, self.population_size // 10)
                parents = population[:num_parents]
                
                next_generation = []
                next_generation.extend(parents) # Elitism: The absolute best bugs survive unchanged
                
                while len(next_generation) < self.population_size:
                    # Pick two parents randomly from the elite pool
                    parent_a = random.choice(parents)
                    parent_b = random.choice(parents)
                    
                    # They breed!
                    child_bug = parent_a.spawn_child(
                        mutation_rate=self.mutation_rate, 
                        other_parent=parent_b
                    )
                    
                    next_generation.append(child_bug)
                    
                population = next_generation

        Log.info(f"\n--- {bug_name} EVOLUTION COMPLETE ---")
        return population[0] # Return the absolute best bug from the final generation

#
# Fitness Functions
#
# What does it mean to "win"?
#

def fitness_gluttony(world, turns_survived):
    """
    The 'Yo, food is good' mindset.
    Ignores how many turns it took, solely rewards the amount of food eaten.
    Pure greed - only cares about feast quantity.
    """
    return float(getattr(world, 'food_collected', 0))

def fitness_longevity(world, turns_survived):
    """
    The survivalist mindset.
    Rewards staying alive as long as possible. Food is only a means to an end.
    Pure endurance - maximize lifespan regardless of consumption.
    """
    return float(turns_survived)

def fitness_efficiency(world, turns_survived):
    """
    A hybrid mindset.
    Rewards eating food, but PENALIZES taking too long to do it. 
    Balanced approach - food matters, but speed matters more.
    """
    food = getattr(world, 'food_collected', 0)
    # 50 points per food, minus 1 point for every turn wasted
    return (food * 50.0) - turns_survived

def fitness_speed_raider(world, turns_survived):
    """
    The aggressive hunter.
    Heavily rewards early food consumption - first meal is CRITICAL.
    If no food found, penalty scales over time.
    Specializes in resource-dense environments and direct paths.
    """
    food = getattr(world, 'food_collected', 0)
    if food == 0:
        # Severe penalty for never finding food
        return -turns_survived
    # Exponential reward for eating fast (first food most valuable)
    # Each food eaten at turn T gets: 100 / sqrt(T)
    # This heavily favors finding food quickly
    return (food * 100.0) - (turns_survived * 0.5)

def fitness_sustenance(world, turns_survived):
    """
    The steady forager.
    Rewards consistent food-finding over time. The rate of food consumption matters.
    (food / turns_survived) gives consumption rate - bugs that find food regularly win.
    Penalizes both starvation AND wandering without eating.
    """
    food = getattr(world, 'food_collected', 0)
    if turns_survived == 0:
        return 0.0
    consumption_rate = food / float(turns_survived)
    # Scale by turns survived to prefer bugs that both eat AND survive
    return (consumption_rate * 1000.0) + (turns_survived * 0.1)

def fitness_balanced(world, turns_survived):
    """
    The Swiss Army knife.
    Rewards a balanced approach: need both food AND longevity equally.
    Geometric mean of (food * 50) and turns_survived.
    Great all-rounder, but not specialized in anything.
    """
    food = getattr(world, 'food_collected', 0)
    food_score = max(1, food * 50.0)  # Avoid log(0)
    survival_score = max(1, float(turns_survived))
    # Geometric mean - punishes imbalance
    return (food_score * survival_score) ** 0.5

def fitness_minimalist(world, turns_survived):
    """
    The ascetic philosopher.
    Rewards LONGEVITY while finding ANY food at all.
    Gets a baseline for surviving, bonus ONLY if food is found.
    Bugs that starve get turns_survived/2. Bugs that eat get bonus.
    """
    food = getattr(world, 'food_collected', 0)
    base_score = turns_survived / 2.0
    food_bonus = food * 30.0
    return base_score + food_bonus

def fitness_feast_or_famine(world, turns_survived):
    """
    The risk-taker's paradox.
    Extreme risk-reward: massive bonus for finding food quickly, harsh penalty for starvation.
    Only food count matters - turns are a tiebreaker.
    Winners: bugs that find ANY food (even 1) survive. Losers: bugs that find nothing.
    Creates extreme specialization pressure.
    """
    food = getattr(world, 'food_collected', 0)
    if food == 0:
        return turns_survived - 100.0  # Severe starvation penalty
    # Every food is worth a LOT - makes specialization attractive
    return food * 200.0

if __name__ == "__main__":
    #
    # Training Config
    #
    fitness_fns = [
        fitness_efficiency, 
        fitness_gluttony, 
        fitness_longevity,
        fitness_speed_raider,
        fitness_sustenance,
        fitness_balanced,
        fitness_minimalist,
        fitness_feast_or_famine
    ]

    MAP_CREATION_TYPE = "dungeon"

    for fitness_fn in fitness_fns:
        for name, vision in VISION_CONES.items():
            print("-"*10)
            print(f"Vision: {name} - {fitness_fn.__name__}")
            print("-"*10)

            neural_path  = f'bug_saves/neural-{name}-{fitness_fn.__name__}-dungeon.json'
            memory_path  = f"bug_saves/memory-{name}-{fitness_fn.__name__}-dungeon.json"
            torchnn_path = f"bug_saves/torchnn-{name}-{fitness_fn.__name__}-dungeon.json"

            if not os.path.exists(neural_path):
                trainer_neural = EvolutionaryTrainer(
                    bug_class=NeuralBug,
                    vision_cone=vision,
                    generations=GENERATIONS,
                    population_size=POP_SIZE,
                    mutation_rate=MUTATION_RATE,
                    fitness_fn=fitness_fn,
                    map_layout=MAP_CREATION_TYPE,
                    trials=TRIALS_PER_EPOCH,
                    name="Neural"
                )
                best_neural = trainer_neural.train()
                best_neural.save_to_file(neural_path)

            if not os.path.exists(memory_path):
                trainer_memory = EvolutionaryTrainer(
                    bug_class=MemoryBug,
                    vision_cone=vision,
                    fitness_fn=fitness_fn,
                    generations=GENERATIONS,
                    population_size=POP_SIZE,
                    mutation_rate=MUTATION_RATE,
                    map_layout=MAP_CREATION_TYPE,
                    trials=TRIALS_PER_EPOCH,
                    name="Memory"
                )
                best_memory = trainer_memory.train()
                best_memory.save_to_file(memory_path)

            if not os.path.exists(torchnn_path):
                trainer_torchnn = EvolutionaryTrainer(
                    bug_class=TorchBug,
                    vision_cone=vision,
                    fitness_fn=fitness_fn,
                    generations=GENERATIONS,
                    population_size=POP_SIZE,
                    mutation_rate=MUTATION_RATE,
                    map_layout=MAP_CREATION_TYPE,
                    trials=TRIALS_PER_EPOCH,
                    name="TorchNN"
                )
                best_torchnn = trainer_torchnn.train()
                best_torchnn.save_to_file(torchnn_path)

            print("")
            print("")