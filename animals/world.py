from dataclasses import dataclass
from typing import Optional
import torch
import math


@dataclass
class SensorCone:
    fov_deg: int     = 120
    front_radius: int   = 5
    side_radius: int    = 5

@dataclass
class SensorRequest:
    """
    The request from an entity to the World to read information. Each
    "type" of sensor can have its own cone.
    """
    vision: SensorCone

#
# Map/Movement Helpers
#
# Conventions: 
#   - dy is row offest, dx is column offset
#   - "facing up" means forward = decreaase row index
#   - angle 0deg = straight up
#

#
# MOVEMENT CONSTS
#
FORWARD = [-1, 0]
RIGHT = [0, 1]
DOWN = [1, 0]
LEFT = [0, -1]

def generate_cone_offsets(cone: SensorCone) -> torch.Tensor:
    max_r = max(cone.front_radius, cone.side_radius)
    offsets = []
    half_fov = math.radians(cone.fov_deg / 2)

    for dy in range(-max_r, max_r + 1):
        for dx in range(-max_r, max_r + 1):
            if dx == 0 and dy == 0:
                continue  # skip self-cell, or include it if you want self always visible

            # ellipse check (front_radius governs forward extent, side_radius lateral)
            ellipse = (dx / cone.side_radius) ** 2 + (dy / cone.front_radius) ** 2
            if ellipse > 1.0:
                continue

            # angle check: 0 = straight ahead (up = -y direction)
            angle = math.atan2(dx, -dy)
            if abs(angle) > half_fov:
                continue

            offsets.append((dy, dx))

    return torch.tensor(offsets, dtype=torch.long)  # shape (N, 2)

def rotate_offsets_90(offsets: torch.Tensor) -> torch.Tensor:
    dy, dx = offsets[:, 0], offsets[:, 1]
    return torch.stack([dx, -dy], dim=1)

def generate_offsets_by_heading(cone: SensorCone) -> torch.Tensor:
    base = generate_cone_offsets(cone)
    all_headings = [base]
    current = base
    for _ in range(3):
        current = rotate_offsets_90(current)
        all_headings.append(current)
    return torch.stack(all_headings, dim=0)  # shape (4, N, 2)

def _bresenham_intermediate(dy: int, dx: int):
    """
    Return the list of (dy, dx) cells strictly between (0,0) and (dy,dx),
    NOT including either endpoint, using Bresenham's line algorithm.
    """
    points = []
    x1, y1 = dx, dy

    cx, cy = 0, 0
    sx = 1 if x1 > 0 else -1
    sy = 1 if y1 > 0 else -1
    adx, ady = abs(x1), abs(y1)
    err = adx - ady

    while (cx, cy) != (x1, y1):
        e2 = 2 * err
        if e2 > -ady:
            err -= ady
            cx += sx
        if e2 < adx:
            err += adx
            cy += sy
        if (cx, cy) != (x1, y1):
            points.append((cy, cx))  # (dy, dx)

    return points

def generate_ray_offsets(offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each (dy, dx) in offsets, compute the cells strictly between (0,0)
    and (dy, dx), padded to a common length with (0,0) (the bug's own cell,
    which is always safe/non-wall, so padding never falsely blocks).

    Returns:
        ray_offsets: (N, max_len, 2) padded intermediate cell offsets
        ray_lens:    (N,) actual number of intermediate cells for each offset
    """
    all_points = [_bresenham_intermediate(int(dy), int(dx)) for dy, dx in offsets.tolist()]
    max_len = max((len(p) for p in all_points), default=0)
    max_len = max(max_len, 1)  # avoid zero-length tensors

    ray_offsets = torch.zeros((len(all_points), max_len, 2), dtype=torch.long)
    ray_lens = torch.zeros((len(all_points),), dtype=torch.long)

    for i, pts in enumerate(all_points):
        ray_lens[i] = len(pts)
        for j, (py, px) in enumerate(pts):
            ray_offsets[i, j, 0] = py
            ray_offsets[i, j, 1] = px

    return ray_offsets, ray_lens

def generate_ray_offsets_by_heading(offsets_by_heading: torch.Tensor) -> torch.Tensor:
    """
    offsets_by_heading: (4, N, 2) -- the per-heading cone offsets.
    Returns ray_offsets_by_heading: (4, N, max_len, 2)
    The mask (ray_lens) is the same for every heading since rotation by 90deg
    preserves line length, so we only need to return one mask.
    """
    all_rays = []
    ray_lens = None
    for h in range(offsets_by_heading.shape[0]):
        rays, lens = generate_ray_offsets(offsets_by_heading[h])
        all_rays.append(rays)
        if ray_lens is None:
            ray_lens = lens
    max_len = max(r.shape[1] for r in all_rays)
    # pad all to same max_len
    padded = []
    for r in all_rays:
        if r.shape[1] < max_len:
            pad = torch.zeros((r.shape[0], max_len - r.shape[1], 2), dtype=torch.long)
            r = torch.cat([r, pad], dim=1)
        padded.append(r)
    return torch.stack(padded, dim=0), ray_lens  # (4, N, max_len, 2), (N,)

@dataclass
class BiomeConfig:
    """
    Configuration for how food is generated in an area of the map
    """
    x: int # top left x of where area starts
    y: int # top left y of where are starts
    width: int # how big the area is in widdth
    height: int # how big the area is in height
    food_refresh_rate: float # how often per tick missing food can spawn here
    eating_bonus: float # how much each food gives in health

@dataclass
class WorldConfig:
    """
    The configuration needed to create a new World class
    """
    bug_sensors: SensorRequest # we only take one for now
    biomes: list[BiomeConfig]
    max_life_force: float = 100.0
    grid_size: int = 24 # X by X grid is the world
    envs: int = 1 # number of parallel worlds to run
    device: str = 'cuda'
    num_bugs: int = 1 # this makes things easier for now
    min_food: int = 15 # The exact amount of food to guarantee per environment
    num_rooms: int = 5          # Number of rooms per environment
    min_room_size: int = 4      # Minimum width/height of a room
    max_room_size: int = 8      # Maximum width/height of a room

class World:
    # Values for what we have in our world
    EMPTY = 0
    WALL = -1
    FOOD = 1
    ANIMAL = 2

    # Values for allowed Movement Actions
    FORWARD = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg

        cone = self.cfg.bug_sensors.vision
        self.vision_cone = cone
        self.cone_offsets = generate_offsets_by_heading(self.vision_cone).to(self.cfg.device)
        self.obs_size = self.cone_offsets.shape[1] # 'V' visible cells

        # Precompute, for each (heading, visible-cell), the intermediate cells
        # along the line-of-sight ray from the bug to that cell. Used to
        # determine if a wall blocks vision to that cell.
        ray_offsets, ray_lens = generate_ray_offsets_by_heading(self.cone_offsets.cpu())
        self.ray_offsets = ray_offsets.to(self.cfg.device)          # (4, V, max_len, 2)
        self.ray_len_mask = (
            torch.arange(ray_offsets.shape[2], device=self.cfg.device).unsqueeze(0)
            < ray_lens.to(self.cfg.device).unsqueeze(1)
        )  # (V, max_len) bool, True where that ray step is "real" (not padding)

        self.heading_offset = torch.tensor([
            FORWARD,
            RIGHT,
            DOWN,
            LEFT,
        ], device=self.cfg.device)
    
    def _setup_biomes(self):
        # Create static tensors to hold the properties for each grid cell
        self.food_refresh_map = torch.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=torch.float32, device=self.cfg.device)
        self.eating_bonus_map = torch.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=torch.float32, device=self.cfg.device)

        for biome in self.cfg.biomes:
            # Assuming y is rows (vertical) and x is columns (horizontal)
            y_start, y_end = biome.y, biome.y + biome.height
            x_start, x_end = biome.x, biome.x + biome.width

            # Apply the biome properties to the specified slices
            self.food_refresh_map[y_start:y_end, x_start:x_end] = biome.food_refresh_rate
            self.eating_bonus_map[y_start:y_end, x_start:x_end] = biome.eating_bonus

    def _generate_map(self, layout: str = "hard"):
        """
        Generates the map according to a difficulty "layout":
          - "easy":   one big open room (just the outer walls). Great for
                      sanity-checking that the bug can find food/move at all.
          - "medium": a handful of large rooms connected by corridors.
          - "hard":   the original densely-packed small-room dungeon
                      (uses cfg.num_rooms / min_room_size / max_room_size).

        Biome placement (where food spawns/refreshes) is independent of the
        wall layout -- it's still driven entirely by cfg.biomes, so the same
        biome config can be tested against easy/medium/hard layouts.
        """
        self.layout = layout

        if layout == "easy":
            self._generate_open_map()
        elif layout == "medium":
            self._generate_rooms_map(
                num_rooms=3,
                min_room_size=max(8, self.cfg.grid_size // 3),
                max_room_size=max(10, self.cfg.grid_size // 2),
            )
        elif layout == "hard":
            self._generate_rooms_map(
                num_rooms=self.cfg.num_rooms,
                min_room_size=self.cfg.min_room_size,
                max_room_size=self.cfg.max_room_size,
            )
        else:
            raise ValueError(f"Unknown map layout '{layout}'. Expected 'easy', 'medium', or 'hard'.")

        self._setup_biomes()

    def _generate_open_map(self):
        """
        "Easy" preset: a single open room bordered by walls. No corridors,
        no obstacles -- just the bug, the biomes, and open floor.
        """
        self.map = torch.full(
            (self.cfg.envs, self.cfg.grid_size, self.cfg.grid_size),
            self.EMPTY, dtype=torch.long, device=self.cfg.device,
        )

        # Outer border walls
        self.map[:, 0, :] = self.WALL
        self.map[:, -1, :] = self.WALL
        self.map[:, :, 0] = self.WALL
        self.map[:, :, -1] = self.WALL

    def _generate_rooms_map(self, num_rooms: int, min_room_size: int, max_room_size: int):
        """
        Generates a batched procedural dungeon map using tensor broadcasting.
        Starts with solid walls, punches out `num_rooms` rooms (sized between
        min_room_size and max_room_size), and connects them with corridors.

        This is the generic room/corridor generator used by both the
        "medium" and "hard" presets -- only the room count/size ranges
        differ between them.
        """
        # Guard against degenerate ranges (e.g. medium preset on a tiny grid)
        min_room_size = max(2, min(min_room_size, self.cfg.grid_size - 2))
        max_room_size = max(min_room_size + 1, min(max_room_size, self.cfg.grid_size - 1))

        # 1. Fill the entire map with walls
        self.map = torch.full((self.cfg.envs, self.cfg.grid_size, self.cfg.grid_size), 
                              self.WALL, dtype=torch.long, device=self.cfg.device)

        # 2. Generate random room dimensions and coordinates for all envs simultaneously
        # Shape: (envs, num_rooms)
        rooms_w = torch.randint(min_room_size, max_room_size, 
                                (self.cfg.envs, num_rooms), device=self.cfg.device)
        rooms_h = torch.randint(min_room_size, max_room_size, 
                                (self.cfg.envs, num_rooms), device=self.cfg.device)
        
        # Ensure rooms don't spawn out of bounds (padding by 1 to keep outer walls intact)
        rooms_x = torch.randint(1, self.cfg.grid_size - max_room_size, 
                                (self.cfg.envs, num_rooms), device=self.cfg.device)
        rooms_y = torch.randint(1, self.cfg.grid_size - max_room_size, 
                                (self.cfg.envs, num_rooms), device=self.cfg.device)

        # 3. Create a 2D coordinate grid for the whole map
        yy, xx = torch.meshgrid(torch.arange(self.cfg.grid_size, device=self.cfg.device), 
                                torch.arange(self.cfg.grid_size, device=self.cfg.device), indexing='ij')
        
        # Reshape for 4D broadcasting: (1, 1, grid_size, grid_size)
        xx = xx.view(1, 1, self.cfg.grid_size, self.cfg.grid_size)
        yy = yy.view(1, 1, self.cfg.grid_size, self.cfg.grid_size)

        # Reshape room boundaries for 4D broadcasting: (envs, num_rooms, 1, 1)
        rx = rooms_x.view(self.cfg.envs, num_rooms, 1, 1)
        ry = rooms_y.view(self.cfg.envs, num_rooms, 1, 1)
        rw = rooms_w.view(self.cfg.envs, num_rooms, 1, 1)
        rh = rooms_h.view(self.cfg.envs, num_rooms, 1, 1)

        # 4. Punch out the rooms
        # Build a mask of all spaces that fall inside ANY room
        room_mask = (xx >= rx) & (xx < rx + rw) & (yy >= ry) & (yy < ry + rh)
        
        # Collapse the num_rooms dimension to get the final 2D mask per env
        is_room = room_mask.any(dim=1)
        self.map[is_room] = self.EMPTY

        # 5. Calculate Room Centers for Corridors
        centers_x = rooms_x + (rooms_w // 2)
        centers_y = rooms_y + (rooms_h // 2)

        # Squeeze the extra dimension out for the corridors ---
        # Shape becomes: (1, grid_size, grid_size)
        xx_3d = xx.squeeze(1)
        yy_3d = yy.squeeze(1)

        # 6. Carve L-shaped Corridors to connect room `i` to room `i+1`
        for i in range(num_rooms - 1):
            start_x = centers_x[:, i].view(-1, 1, 1)
            start_y = centers_y[:, i].view(-1, 1, 1)
            end_x = centers_x[:, i + 1].view(-1, 1, 1)
            end_y = centers_y[:, i + 1].view(-1, 1, 1)

            min_x = torch.minimum(start_x, end_x)
            max_x = torch.maximum(start_x, end_x)
            min_y = torch.minimum(start_y, end_y)
            max_y = torch.maximum(start_y, end_y)

            # Use the newly squeezed 3D coordinates!
            h_corridor = (yy_3d == start_y) & (xx_3d >= min_x) & (xx_3d <= max_x)
            v_corridor = (xx_3d == end_x) & (yy_3d >= min_y) & (yy_3d <= max_y)

            # Mask is now perfectly (envs, grid_size, grid_size)
            self.map[h_corridor | v_corridor] = self.EMPTY

        # 7. Safety Net: Enforce strict outer borders just in case
        self.map[:, 0, :] = self.WALL
        self.map[:, -1, :] = self.WALL
        self.map[:, :, 0] = self.WALL
        self.map[:, :, -1] = self.WALL


    def _generate_positions(self):
        """
        Spawns bugs anywhere on EMPTY ground, EXCLUDING biome boundaries
        (so bugs don't start standing right on top of food zones).
        """
        # 1. Start with uniform probability everywhere
        # Shape: (grid_size, grid_size)
        prob_map = torch.ones((self.cfg.grid_size, self.cfg.grid_size), device=self.cfg.device)
 
        # 2. Zero out probability in all biome 
        for biome in self.cfg.biomes:
            prob_map[biome.y : biome.y + biome.height, 
                     biome.x : biome.x + biome.width] = 0.0
 
        # 3. Mask out walls: never spawn on a wall
        is_wall = (self.map == self.WALL)
        prob_map[is_wall[0]] = 0.0
 
        # 4. Flatten the map to 1D so we can use multinomial sampling
        flat_probs = prob_map.view(-1)
        
        # 5. Sample random indices for all bugs across all environments
        # total_bugs = envs * num_bugs
        total_bugs = self.cfg.envs * self.cfg.num_bugs
        
        # We sample indices from the probability map
        flat_indices = torch.multinomial(flat_probs, num_samples=total_bugs, replacement=True)
        
        # 6. Convert 1D indices back to 2D (row, col)
        spawn_rows = flat_indices // self.cfg.grid_size
        spawn_cols = flat_indices % self.cfg.grid_size
 
        # 7. Reshape and assign to positions
        # self.positions shape: (envs, num_bugs, 2)
        self.positions = torch.stack([spawn_rows, spawn_cols], dim=-1).view(self.cfg.envs, self.cfg.num_bugs, 2)
        
        # 8. Mark these positions as ANIMAL on the map
        batch_indices = torch.arange(self.cfg.envs, device=self.cfg.device).view(-1, 1).expand(-1, self.cfg.num_bugs)
        self.map[batch_indices, self.positions[..., 0], self.positions[..., 1]] = self.ANIMAL
 
        # 9. Initialize headings (0: Up, 1: Right, 2: Down, 3: Left)
        self.headings = torch.randint(0, 4, (self.cfg.envs, self.cfg.num_bugs), device=self.cfg.device)
 
        # Give every bug a starting life force (e.g., 100.0)
        self.life_force = torch.full((self.cfg.envs, self.cfg.num_bugs), self.cfg.max_life_force, dtype=torch.float32, device=self.cfg.device)
    def _populate_initial_food(self):
        """
        Instantly populates all environments with exactly `min_food` based on biome probabilities.
        """
        # 1. Expand the 2D biome probability map to 3D to match all environments
        base_probs = self.food_refresh_map.expand(self.cfg.envs, -1, -1).clone()

        # 2. Mask out walls, bugs, or anything that isn't an EMPTY space
        empty_spaces = (self.map == self.EMPTY)
        base_probs[~empty_spaces] = 0.0

        # 3. Flatten the probabilities to 1D arrays for multinomial
        flat_probs = base_probs.view(self.cfg.envs, -1)
        
        # Safety fallback: Give all empty spaces a tiny uniform chance just in case 
        # the biome is physically too small to hold `min_food`
        flat_probs += (empty_spaces.view(self.cfg.envs, -1) * 1e-6)

        # 4. Spin the wheel! Grab `min_food` unique spots per environment
        # replacement=False guarantees we don't spawn two foods on the exact same tile
        spawn_indices = torch.multinomial(flat_probs, num_samples=self.cfg.min_food, replacement=False)

        # 5. Convert the 1D indices back to 2D rows and columns
        spawn_rows = spawn_indices // self.cfg.grid_size
        spawn_cols = spawn_indices % self.cfg.grid_size
        
        # 6. Create a matching tensor of environment indices
        env_indices = torch.arange(self.cfg.envs, device=self.cfg.device).view(-1, 1).expand(-1, self.cfg.min_food)

        # 7. Draw the food onto the map
        self.map[env_indices, spawn_rows, spawn_cols] = self.FOOD        
    
    def _spawn_food(self):
        # 1. Generate a random float between 0.0 and 1.0 for every cell in every environment
        # Shape: (envs, grid_size, grid_size)
        rand_spawns = torch.rand((self.cfg.envs, self.cfg.grid_size, self.cfg.grid_size), device=self.cfg.device)

        # 2. Identify valid empty spaces where food CAN spawn (e.g., not a wall, not a bug, no existing food)
        empty_spaces = (self.map == self.EMPTY) 

        # 3. Create a mask: cell is empty AND the random chance hit the refresh rate
        # self.food_refresh_map automatically broadcasts from (grid, grid) to (envs, grid, grid)
        spawn_mask = empty_spaces & (rand_spawns < self.food_refresh_map)

        # 4. Apply the food to the map
        self.map[spawn_mask] = self.FOOD
    
    def _spawn_food(self):
        # 1. Generate a random float between 0.0 and 1.0 for every cell in every environment
        # Shape: (envs, grid_size, grid_size)
        rand_spawns = torch.rand((self.cfg.envs, self.cfg.grid_size, self.cfg.grid_size), device=self.cfg.device)

        # 2. Identify valid empty spaces where food CAN spawn (e.g., not a wall, not a bug, no existing food)
        empty_spaces = (self.map == self.EMPTY) 

        # 3. Create a mask: cell is empty AND the random chance hit the refresh rate
        # self.food_refresh_map automatically broadcasts from (grid, grid) to (envs, grid, grid)
        spawn_mask = empty_spaces & (rand_spawns < self.food_refresh_map)

        # 4. Apply the food to the map
        self.map[spawn_mask] = self.FOOD

    def reset(self, layout: str = "easy"):
        """
        layout: "easy" | "medium" | "hard" -- controls map generation
        difficulty (see _generate_map for details). Biomes (food locations)
        are unaffected and still come from cfg.biomes.
        """
        self._generate_map(layout)
        self._generate_positions()
        
        self._populate_initial_food()
        self.food_eaten = torch.zeros(self.cfg.envs, dtype=torch.long, device=self.cfg.device)
        # Return the very first observation of the fresh environment
        return self.get_observations()

    def get_observations(self):
        """
        Returns observations for all bugs. Shape: (envs, num_bugs, V)

        Cells hidden behind a wall (i.e. any cell along the line-of-sight
        between the bug and the target cell is a WALL) are reported as WALL,
        regardless of what's actually at the target cell.
        """
        bug_offsets = self.cone_offsets[self.headings]  # (envs, num_bugs, V, 2)

        bug_rows = self.positions[..., 0].unsqueeze(2)
        bug_cols = self.positions[..., 1].unsqueeze(2)

        obs_rows = torch.clamp(bug_rows + bug_offsets[..., 0], 0, self.cfg.grid_size - 1)
        obs_cols = torch.clamp(bug_cols + bug_offsets[..., 1], 0, self.cfg.grid_size - 1)

        env_indices = torch.arange(self.cfg.envs, device=self.cfg.device).view(-1, 1, 1).expand(-1, self.cfg.num_bugs, self.obs_size)

        target_vals = self.map[env_indices, obs_rows, obs_cols]  # (envs, num_bugs, V)

        # --- Line-of-sight check ---
        # ray_offsets_for_heading: (envs, num_bugs, V, max_len, 2)
        ray_offsets_for_heading = self.ray_offsets[self.headings]

        ray_rows = torch.clamp(
            bug_rows.unsqueeze(-1) + ray_offsets_for_heading[..., 0], 0, self.cfg.grid_size - 1
        )  # (envs, num_bugs, V, max_len)
        ray_cols = torch.clamp(
            bug_cols.unsqueeze(-1) + ray_offsets_for_heading[..., 1], 0, self.cfg.grid_size - 1
        )

        env_indices_ray = env_indices.unsqueeze(-1).expand_as(ray_rows)
        ray_vals = self.map[env_indices_ray, ray_rows, ray_cols]  # (envs, num_bugs, V, max_len)

        # mask out padding ray steps (treat as non-wall so they never block)
        is_wall_along_ray = (ray_vals == self.WALL) & self.ray_len_mask.view(1, 1, *self.ray_len_mask.shape)
        blocked = is_wall_along_ray.any(dim=-1)  # (envs, num_bugs, V)

        return torch.where(blocked, torch.tensor(self.WALL, device=self.cfg.device), target_vals)
    
    def step(self, actions: torch.Tensor):
        """
        actions shape: (envs, num_bugs)
        """
        # Making a step costs food. We allow eating and not dying this turn later
        initial_life = self.life_force.clone()
        self.life_force -= 1.0

        # 1. Update self.positions and self.headings based on actions
        # create a boolean mask for all actions that are for right
        right_mask = (actions == self.RIGHT)
        left_mask = (actions == self.LEFT)


        # Apply those boolean masks correctly for heading
        self.headings[right_mask] = (self.headings[right_mask] + 1) % 4 # mod 4 so we get 3 + 1 = 0
        self.headings[left_mask] = (self.headings[left_mask] - 1) % 4

        # Calculate movement
        step_offset = self.heading_offset[self.headings]
        target_positions = self.positions + step_offset

        forward_mask = (actions == self.FORWARD)
        batch_env_indices = torch.arange(self.cfg.envs, device=self.cfg.device).view(-1, 1).expand(-1, self.cfg.num_bugs)

        # Extract intended row/col, clamped to grid edges just in case they aim out of bounds
        target_rows = torch.clamp(target_positions[..., 0], 0, self.cfg.grid_size - 1)
        target_cols = torch.clamp(target_positions[..., 1], 0, self.cfg.grid_size - 1)

        # Check the map at the target locations
        target_cell_values = self.map[batch_env_indices, target_rows, target_cols]

        # A move is valid if they wanted to go forward AND the target is not a wall
        valid_move_mask = forward_mask & (target_cell_values != self.WALL)
        
        # Eating food
        ate_food_mask = valid_move_mask & (target_cell_values == self.FOOD)

        # Extract the exact map coordinates for the bugs that ate
        food_rows = target_rows[ate_food_mask]
        food_cols = target_cols[ate_food_mask]
    
        # Extract flat lists of ONLY the bugs that are allowed to move
        moving_envs = batch_env_indices[valid_move_mask]
        old_rows = self.positions[..., 0][valid_move_mask]
        old_cols = self.positions[..., 1][valid_move_mask]
        new_rows = target_rows[valid_move_mask]
        new_cols = target_cols[valid_move_mask]

        # Execute the move on the map and update state
        self.map[moving_envs, old_rows, old_cols] = self.EMPTY
        self.positions[valid_move_mask] = target_positions[valid_move_mask]
        self.map[moving_envs, new_rows, new_cols] = self.ANIMAL

        if ate_food_mask.any():
            # Extract coordinates and apply life force bonuses
            food_rows = target_rows[ate_food_mask]
            food_cols = target_cols[ate_food_mask]
            bonuses = self.eating_bonus_map[food_rows, food_cols]

            self.life_force[ate_food_mask] = torch.clamp(
                self.life_force[ate_food_mask] + bonuses, 
                max=self.cfg.max_life_force
            )

            # Get a 1D list of exactly which environments need a replacement food
            envs_that_ate = batch_env_indices[ate_food_mask]
            envs_that_ate = batch_env_indices[ate_food_mask]
            
            ones = torch.ones_like(envs_that_ate, dtype=self.food_eaten.dtype)
            self.food_eaten.index_add_(0, envs_that_ate, ones)

            #  Build a probability map for these specific environments
            # Expand the 2D biome map to match the number of foods we need to spawn
            base_probs = self.food_refresh_map.expand(len(envs_that_ate), -1, -1).clone()
            
            #  Mask out invalid spaces
            # We only want to spawn on EMPTY tiles. 
            empty_in_envs = (self.map[envs_that_ate] == self.EMPTY)
            base_probs[~empty_in_envs] = 0.0 # Force non-empty spaces to 0% chance
            
            #  Flatten the 2D grid to a 1D array so multinomial can process it
            flat_probs = base_probs.view(len(envs_that_ate), -1)
            
            # Safety fallback: If a biome is completely full and has 0 empty spaces, 
            # give a tiny uniform chance to all empty spaces on the map to prevent a crash
            flat_probs += (empty_in_envs.view(len(envs_that_ate), -1) * 1e-6)

            #  Spin the weighted roulette wheel! Grab 1 random index per eaten food
            new_flat_indices = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            
            #  Convert the 1D index back to 2D (row, col) coordinates
            new_rows = new_flat_indices // self.cfg.grid_size
            new_cols = new_flat_indices % self.cfg.grid_size
            
            #  Plop the new food onto the map
            self.map[envs_that_ate, new_rows, new_cols] = self.FOOD
        
        dones = (self.life_force <= 0)
        
        # Rewards calculate cleanly
        rewards = torch.ones_like(self.life_force, dtype=torch.float32, device=self.cfg.device)
        rewards[dones] = -1

        dead_rows = self.positions[..., 0][dones]
        dead_cols = self.positions[..., 1][dones]
        self.map[batch_env_indices[dones], dead_rows, dead_cols] = self.EMPTY
        
        #  Track who still needs a respawn
        needs_respawn = dones.clone()

        #  Loop until every dead bug has successfully found an EMPTY spot
        while needs_respawn.any():
            rand_rows = torch.randint(1, self.cfg.grid_size - 1, (self.cfg.envs, self.cfg.num_bugs), device=self.cfg.device)
            rand_cols = torch.randint(1, self.cfg.grid_size - 1, (self.cfg.envs, self.cfg.num_bugs), device=self.cfg.device)

            target_cells = self.map[batch_env_indices, rand_rows, rand_cols]
            
            # Valid if it needs a respawn AND the random spot is empty
            valid_spot = needs_respawn & (target_cells == self.EMPTY)
            
            # Apply successful coordinates
            self.positions[..., 0] = torch.where(valid_spot, rand_rows, self.positions[..., 0])
            self.positions[..., 1] = torch.where(valid_spot, rand_cols, self.positions[..., 1])
            
            # Draw the successfully respawned bugs onto the map
            valid_envs = batch_env_indices[valid_spot]
            valid_rows = rand_rows[valid_spot]
            valid_cols = rand_cols[valid_spot]
            self.map[valid_envs, valid_rows, valid_cols] = self.ANIMAL

            # Unflag them so they don't roll again in the next while loop iteration
            needs_respawn &= ~valid_spot

        # 4. Now that everyone is guaranteed to be on the board, reset life force
        self.life_force[dones] = 100.0

        return self.get_observations(), rewards, dones

    def render(self, env_idx=0, cell_size=24, fps=15, last_action=None):
        """
        Renders a visually stunning, bioluminescent environment with Bug Vision
        and Telemetry properly stacked in a unified side panel.
        """
        import pygame
        import sys

        if not hasattr(self, 'screen'):
            pygame.init()
            self.cell_size = cell_size
            self.panel_width = 320 
            self.grid_pixel_size = self.cfg.grid_size * cell_size
            
            self.width = self.grid_pixel_size + self.panel_width
            self.height = max(self.grid_pixel_size, 550) 
            
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(f"Bioluminescent Bug World - Env {env_idx}")
            self.clock = pygame.time.Clock()
            
            # --- Fonts ---
            try:
                self.font_title = pygame.font.SysFont("segoeui, roboto, arial", 28, bold=True)
                self.font_sm = pygame.font.SysFont("segoeui, roboto, arial", 18)
            except:
                self.font_title = pygame.font.SysFont(None, 36)
                self.font_sm = pygame.font.SysFont(None, 24)
            
            # --- Map Color Palette ---
            self.bg_color = (15, 15, 20)
            self.wall_color = (35, 35, 45)
            self.wall_highlight = (55, 55, 65)
            self.grid_color = (25, 25, 35)
            self.food_base_color = (255, 50, 150)
            self.food_glow_color = (255, 50, 150, 60)
            self.bug_color = (0, 255, 255)

            # --- Panel Color Palette ---
            self.ui_bg = (24, 24, 24)
            self.ui_text_primary = (255, 255, 255)
            self.ui_text_sec = (170, 170, 170)
            self.ui_accent = (255, 193, 7)      
            self.ui_success = (129, 199, 132)   
            self.ui_danger = (229, 115, 115)    
            self.ui_border = (60, 60, 60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        grid = self.map[env_idx].cpu().numpy()
        self.screen.fill(self.bg_color)

        # ---------------------------------------------------------
        # 1. DRAW THE MAIN WORLD
        # ---------------------------------------------------------
        for row in range(self.cfg.grid_size):
            for col in range(self.cfg.grid_size):
                val = grid[row, col]
                x = col * self.cell_size
                y = row * self.cell_size
                
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.grid_color, rect, 1)

                if val == self.WALL:
                    pygame.draw.rect(self.screen, self.wall_color, rect)
                    inner_rect = pygame.Rect(x + 2, y + 2, self.cell_size - 4, self.cell_size - 4)
                    pygame.draw.rect(self.screen, self.wall_highlight, inner_rect, 1, border_radius=2)

                elif val == self.FOOD:
                    center = (x + self.cell_size // 2, y + self.cell_size // 2)
                    glow_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, self.food_glow_color, (self.cell_size//2, self.cell_size//2), self.cell_size // 2.5)
                    self.screen.blit(glow_surf, (x, y))
                    pygame.draw.circle(self.screen, self.food_base_color, center, self.cell_size // 6)
        
        for biome in self.cfg.biomes:
            bx = biome.x * self.cell_size
            by = biome.y * self.cell_size
            b_width = biome.width * self.cell_size
            b_height = biome.height * self.cell_size
            biome_rect = pygame.Rect(bx, by, b_width, b_height)
            pygame.draw.rect(self.screen, (50, 180, 50), biome_rect, 2, border_radius=4)

        # ---------------------------------------------------------
        # 2. DRAW THE VISION CONE HIGHLIGHTS & BUG (Main Map)
        # ---------------------------------------------------------
        vision_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        vision_surf.fill((0, 255, 255, 15)) 
        
        bug_row, bug_col, heading = 0, 0, 0
        is_alive = self.life_force[env_idx, 0].item() > 0

        if is_alive:
            bug_row = self.positions[env_idx, 0, 0].item()
            bug_col = self.positions[env_idx, 0, 1].item()
            heading = self.headings[env_idx, 0].item()
            
            offsets = self.cone_offsets[heading]
            for i in range(offsets.shape[0]):
                obs_row = bug_row + offsets[i, 0].item()
                obs_col = bug_col + offsets[i, 1].item()
                if 0 <= obs_row < self.cfg.grid_size and 0 <= obs_col < self.cfg.grid_size:
                    self.screen.blit(vision_surf, (obs_col * self.cell_size, obs_row * self.cell_size))

            center_x = bug_col * self.cell_size + self.cell_size / 2
            center_y = bug_row * self.cell_size + self.cell_size / 2
            offset, back_offset, wing = self.cell_size * 0.4, self.cell_size * 0.3, self.cell_size * 0.35

            if heading == 0:   tips = [(center_x, center_y - offset), (center_x - wing, center_y + back_offset), (center_x + wing, center_y + back_offset)]
            elif heading == 1: tips = [(center_x + offset, center_y), (center_x - back_offset, center_y - wing), (center_x - back_offset, center_y + wing)]
            elif heading == 2: tips = [(center_x, center_y + offset), (center_x + wing, center_y - back_offset), (center_x - wing, center_y - back_offset)]
            else:              tips = [(center_x - offset, center_y), (center_x + back_offset, center_y + wing), (center_x + back_offset, center_y - wing)]

            pygame.draw.polygon(self.screen, self.bug_color, tips)
            pygame.draw.circle(self.screen, (255, 255, 255), tips[0], 2)

        # ---------------------------------------------------------
        # 3. DRAW THE SIDE PANEL (Vision TOP, Telemetry BOTTOM)
        # ---------------------------------------------------------
        panel_rect = pygame.Rect(self.grid_pixel_size, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, self.ui_bg, panel_rect)
        pygame.draw.line(self.screen, self.ui_border, (panel_rect.left, 0), (panel_rect.left, self.height), 2)

        current_y = 25

        # ==================== A. FIRST-PERSON RADAR (TOP) ====================
        self.screen.blit(self.font_title.render("BUG VISION", True, self.ui_text_primary), (panel_rect.left + 25, current_y))
        current_y += 45
        
        if is_alive:
            mini_cell = 18
            mini_grid_size = 11 
            
            radar_x = panel_rect.left + (self.panel_width - (mini_grid_size * mini_cell)) // 2
            mini_map_rect = pygame.Rect(radar_x, current_y, mini_grid_size * mini_cell, mini_grid_size * mini_cell)
            
            pygame.draw.rect(self.screen, (12, 12, 12), mini_map_rect, border_radius=6)
            pygame.draw.rect(self.screen, self.ui_border, mini_map_rect, 2, border_radius=6)

            radar_offsets = self.cone_offsets[0].cpu().numpy()
            actual_offsets = self.cone_offsets[heading].cpu().numpy()
            bug_mini_center = 5 

            for i in range(len(actual_offsets)):
                obs_row = bug_row + actual_offsets[i, 0]
                obs_col = bug_col + actual_offsets[i, 1]

                val = self.WALL 
                if 0 <= obs_row < self.cfg.grid_size and 0 <= obs_col < self.cfg.grid_size:
                    val = grid[obs_row, obs_col]

                # Line-of-sight check
                ray_pts = self.ray_offsets[heading, i].cpu().numpy()
                ray_len = int(self.ray_len_mask[i].sum().item())
                for j in range(ray_len):
                    ry = bug_row + ray_pts[j, 0]
                    rx = bug_col + ray_pts[j, 1]
                    if not (0 <= ry < self.cfg.grid_size and 0 <= rx < self.cfg.grid_size):
                        continue
                    if grid[ry, rx] == self.WALL:
                        val = self.WALL
                        break

                if val == self.WALL:   color = (100, 100, 100)
                elif val == self.FOOD: color = self.ui_success
                else:                  color = (30, 30, 35)

                draw_row = bug_mini_center + radar_offsets[i, 0]
                draw_col = bug_mini_center + radar_offsets[i, 1]
                
                cell_rect = pygame.Rect(mini_map_rect.left + draw_col * mini_cell, mini_map_rect.top + draw_row * mini_cell, mini_cell, mini_cell)
                pygame.draw.rect(self.screen, color, cell_rect, border_radius=2)
                pygame.draw.rect(self.screen, (20, 20, 20), cell_rect, 1, border_radius=2) 

            # Bug fixed pointing UP in center of radar
            radar_center_x = mini_map_rect.left + bug_mini_center * mini_cell + mini_cell // 2
            radar_center_y = mini_map_rect.top + bug_mini_center * mini_cell + mini_cell // 2
            pygame.draw.polygon(self.screen, self.bug_color, [
                (radar_center_x, radar_center_y - 6),
                (radar_center_x - 5, radar_center_y + 4),
                (radar_center_x + 5, radar_center_y + 4)
            ])

            # Push current_y down past the radar grid
            current_y += (mini_grid_size * mini_cell) + 30
        else:
            # If dead, just leave space where the radar would be
            current_y += 200 

        # ==================== B. TELEMETRY (BOTTOM) ====================
        # Draw a subtle separator line
        pygame.draw.line(self.screen, self.ui_border, (panel_rect.left + 25, current_y), (panel_rect.right - 25, current_y), 1)
        current_y += 20

        self.screen.blit(self.font_title.render("TELEMETRY", True, self.ui_text_primary), (panel_rect.left + 25, current_y))
        current_y += 45

        def draw_ui_text(font, label, value_text, value_color, y_pos):
            self.screen.blit(font.render(label, True, self.ui_text_sec), (panel_rect.left + 25, y_pos))
            if value_text:
                self.screen.blit(font.render(value_text, True, value_color), (panel_rect.left + 140, y_pos))

        life = self.life_force[env_idx, 0].item()
        eaten = int(self.food_eaten[env_idx].item())
        
        action_map = {0: "FORWARD", 1: "RIGHT", 2: "LEFT"}
        action_str = action_map.get(last_action, "NONE") if last_action is not None else "NONE"
        
        status_str = "ALIVE" if is_alive else "RESPAWNING"
        status_color = self.ui_success if is_alive else self.ui_danger
        life_color = self.ui_success if life > 30 else self.ui_danger

        draw_ui_text(self.font_sm, "Status:", status_str, status_color, current_y)
        current_y += 30
        draw_ui_text(self.font_sm, "Health:", f"{life:.1f}", life_color, current_y)
        current_y += 30
        draw_ui_text(self.font_sm, "Food Eaten:", f"{eaten}", self.ui_text_primary, current_y)
        current_y += 30
        draw_ui_text(self.font_sm, "Last Action:", action_str, self.ui_accent, current_y)

        pygame.display.flip()
        self.clock.tick(fps)

def get_sensors():
    vision=SensorCone(
        fov_deg=120,
        front_radius=7,
        side_radius=3,
    )

    prey_sensors = SensorRequest(
        vision=vision
    )

    return prey_sensors

if __name__ == '__main__':
    import pygame
    # 1. Setup a small test configuration
    biome = BiomeConfig(x=4, y=4, width=8, height=8, food_refresh_rate=0.1, eating_bonus=30.0)
    cfg = WorldConfig(grid_size=15, envs=1, biomes=[biome], num_bugs=1, device='cpu', bug_sensors=get_sensors()) # Use CPU for simple manual testing
    
    env = World(cfg)
    obs = env.reset()

    print("=== MANUAL CONTROL ENGAGED ===")
    print("UP ARROW: Move Forward")
    print("LEFT ARROW: Turn Left")
    print("RIGHT ARROW: Turn Right")
    print("==============================")
    
    # Render the initial frame before any inputs
    env.render(env_idx=0, fps=30)
    
    running = True
    last_action_taken = None # ADD THIS: Persistent tracker
    
    while running:
        action_chosen = None
        
        # 2. Listen for Keyboard Inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action_chosen = env.FORWARD
                elif event.key == pygame.K_LEFT:
                    action_chosen = env.LEFT
                elif event.key == pygame.K_RIGHT:
                    action_chosen = env.RIGHT
        
        # 3. Only step the environment if a valid key was pressed
        if action_chosen is not None:
            actions = torch.full((cfg.envs, cfg.num_bugs), action_chosen, dtype=torch.long, device=cfg.device)
            obs, rewards, dones = env.step(actions)
            
            # UPDATE THIS: Save the action so it persists on screen
            last_action_taken = action_chosen 
            
            current_life = env.life_force[0, 0].item()
            is_dead = dones[0, 0].item()
            action_name = ["FORWARD", "LEFT", "RIGHT"][action_chosen]
            print(f"Action: {action_name:<8} | Life Force: {current_life:>5.1f} | Status: {'DEAD (Respawning...)' if is_dead else 'Alive'}")
            
        # 4. Render the current state using the persistent tracker
        env.render(env_idx=0, fps=30, last_action=last_action_taken)

    pygame.quit()