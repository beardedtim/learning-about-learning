import torch

class World:
    EMPTY = 0
    WALL = 2
    FOOD = 4

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def __init__(
            self, 
            grid_size=32, 
            sensor_radius=5,
            fov_degrees= 5,
            front_fov_radius=5,
            side_fov_radius=2,
            max_life=100.0,
            food_reward=30.0,
            life_decay=1.0,
            min_food=5, 
            device="cuda"):
        self.device         = device
        self.grid_size      = grid_size
        self.sensor_radius  = sensor_radius
        self.min_food       = min_food
        # Step 1: Pre-compute Rotation Lookup Table
        self.cone_offsets   = self._compute_cone_offsets(fov_degrees=fov_degrees, front_radius=front_fov_radius, side_radius=side_fov_radius)
        self.num_sensors    = len(self.cone_offsets)

        offsets_N = self.cone_offsets
        offsets_E = torch.stack([-self.cone_offsets[:, 1], self.cone_offsets[:, 0]], dim=1)
        offsets_S = torch.stack([-self.cone_offsets[:, 0], -self.cone_offsets[:, 1]], dim=1)
        offsets_W = torch.stack([self.cone_offsets[:, 1], -self.cone_offsets[:, 0]], dim=1)

        self.rotated_offsets = torch.stack([offsets_N, offsets_E, offsets_S, offsets_W])

        # Step 2: Pre-compute occlusion map
        self.los_blockers = self._compute_line_of_sight_blockers()

        # State tensors
        self.num_bugs       = 0
        self.map            = None
        self.positions      = None
        self.headings       = None
        self.masks          = None
        self._obs_buffer    = None

        # Life Force
        self.life_force     = None
        self.MAX_LIFE       = max_life
        self.FOOD_REWARD    = food_reward
        self.LIFE_DECAY     = life_decay

        self.fov_degrees        = fov_degrees
        self.front_fov_radius   = front_fov_radius
        self.side_fov_radius    = side_fov_radius

    def _compute_line_of_sight_blockers(self):
        R = self.sensor_radius
        grid_dim = 2 * R + 1
        center = R
        max_blockers = R * 4

        # Build a map: flat square index -> cone index (-1 if not in cone)
        flat_to_cone = torch.full((grid_dim * grid_dim,), -1, dtype=torch.long)
        for cone_i, (cx, cy) in enumerate(self.cone_offsets.cpu().tolist()):
            tx = int(cx) + center
            ty = int(cy) + center
            flat_to_cone[ty * grid_dim + tx] = cone_i

        # Use cone index 0 as the padding value (safe, always valid)
        # We'll use the center tile as padding — but center isn't in the cone.
        # Instead pad with 0 (first cone tile); blocked tiles that hit padding
        # will just read whatever sensor 0 is, which is fine since we only care
        # if it's a WALL.
        pad_idx = 0
        blocker_indices = torch.full((self.num_sensors, max_blockers), pad_idx, dtype=torch.long)

        def bresenham(x0, y0, x1, y1):
            points = []
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            x, y = x0, y0
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            while True:
                points.append((x, y))
                if x == x1 and y == y1:
                    break
                prev_x, prev_y = x, y
                e2 = 2 * err
                moved_x, moved_y = False, False
                if e2 > -dy:
                    err -= dy
                    x += sx
                    moved_x = True
                if e2 < dx:
                    err += dx
                    y += sy
                    moved_y = True
                if moved_x and moved_y:
                    points.append((prev_x, y))
                    points.append((x, prev_y))
            return points

        for i, (cx, cy) in enumerate(self.cone_offsets.cpu().tolist()):
            tx = int(cx) + center
            ty = int(cy) + center
            path = bresenham(center, center, tx, ty)
            blockers = path[1:-1]

            for step, (bx, by) in enumerate(blockers):
                b_flat = by * grid_dim + bx
                cone_idx = flat_to_cone[b_flat].item()
                if cone_idx >= 0:
                    # Blocker is a cone tile — use its cone index
                    blocker_indices[i, step] = cone_idx
                # else: blocker tile isn't in cone (e.g. behind bug),
                # leave as pad_idx — it won't be a wall so no false occlusion

        return blocker_indices.to(self.device)
    
    def _spawn_food(self, needs_food_mask):
        active_search = needs_food_mask.clone()
        batch_idx = torch.arange(self.num_bugs, device=self.device)

        
        # Use a fixed number of attempts instead of a while loop
        # 3-5 iterations is usually plenty to find an empty spot 
        for _ in range(5): 
            rand_x = torch.randint(1, self.grid_size - 1, (self.num_bugs, 10), device=self.device)
            rand_y = torch.randint(1, self.grid_size - 1, (self.num_bugs, 10), device=self.device)
            
            for i in range(10):
                target_tiles = self.map[batch_idx, rand_y[:, i], rand_x[:, i]]
                is_empty = (target_tiles == self.EMPTY)
                on_bug = (rand_x[:, i] == self.positions[:, 0]) & (rand_y[:, i] == self.positions[:, 1])
                
                valid = active_search & is_empty & ~on_bug
                
                # Apply the food directly using the boolean mask.
                # If 'valid' is entirely False, PyTorch skips this inherently on the GPU.
                self.map[batch_idx[valid], rand_y[valid, i], rand_x[valid, i]] = self.FOOD
                active_search = active_search & ~valid

    def _compute_cone_offsets(self, fov_degrees=120, front_radius=5, side_radius=2):
        # Use the larger radius for the initial square so we don't miss anything
        max_radius = max(front_radius, side_radius)
        
        y, x = torch.meshgrid(
            torch.arange(-max_radius, max_radius + 1, device=self.device),
            torch.arange(-max_radius, max_radius + 1, device=self.device),
            indexing='ij'
        )
        offsets = torch.stack([x.flatten(), y.flatten()], dim=1).float()

        # Angle of each tile from straight ahead (0 = forward, ±90 = sides)
        angles = torch.atan2(offsets[:, 0], -offsets[:, 1])
        half_fov = torch.tensor(fov_degrees / 2 * (3.14159 / 180), device=self.device)

        # How far "to the side" is this tile, as a 0→1 value?
        # 0 = straight ahead, 1 = at the edge of the FOV
        angle_ratio = (angles.abs() / half_fov).clamp(0, 1)

        # Interpolate radius: straight ahead = front_radius, edge = side_radius
        effective_radius = front_radius + (side_radius - front_radius) * angle_ratio

        # Distance filter using the per-tile radius
        distances = torch.sqrt(offsets[:, 0]**2 + offsets[:, 1]**2)
        within_range = distances <= effective_radius

        # Angle filter — still need to be inside the FOV
        within_fov = angles.abs() <= half_fov

        # Exclude self
        not_self = (offsets[:, 0] != 0) | (offsets[:, 1] != 0)

        cone_mask = within_range & within_fov & not_self
        return offsets[cone_mask].long()

    def populate(self, bug_masks_list, wall_density=0.2, force_recreate=False):
        """
        Takes a list of 1d vectors and initializes the population and their private maps.
        """
        self.num_bugs = len(bug_masks_list)

        # Stack individual masks: (BATCH_SIZE, NUM_SENSORS)
        self.masks = torch.stack(bug_masks_list).to(self.device)

        # Randomize initial positions & headings
        self.positions = torch.randint(1, self.grid_size - 1, (self.num_bugs, 2), device=self.device)
        self.headings = torch.randint(self.NORTH, self.WEST + 1, (self.num_bugs,), device=self.device)
        self.life_force = torch.full((self.num_bugs,), self.MAX_LIFE // 2, dtype=torch.float32, device=self.device)

        # Generate the private randomized maps
        if self.map is None or force_recreate:
            self.generate_random_map(wall_density)
            
        
        for _ in range(self.min_food):
            needs_food_mask = torch.ones(self.num_bugs, dtype=torch.bool, device=self.device)
            self._spawn_food(needs_food_mask)
            

    def generate_random_map(self, wall_density=0.2):
        """
        Creates a new stack of randomized dungeons: (BATCH_SIZE, GRID_SIZE, GRID_SIZE)
        """
        # 1. Generate batched random noise
        random_noise = torch.rand((self.num_bugs, self.grid_size, self.grid_size), device=self.device)

        # 2. Convert noise into walls
        self.map = torch.where(
            random_noise < wall_density, 
            torch.tensor(self.WALL, dtype=torch.float32, device=self.device), 
            torch.tensor(self.EMPTY, dtype=torch.float32, device=self.device)
        )

        # 3. Apply border walls to ALL maps simultaneously
        self.map[:, 0, :]  = self.WALL
        self.map[:, -1, :] = self.WALL
        self.map[:, :, 0]  = self.WALL
        self.map[:, :, -1] = self.WALL

        # 4. Clear walls where bugs are currently standing
        batch_idx = torch.arange(self.num_bugs, device=self.device)
        self.map[batch_idx, self.positions[:, 1], self.positions[:, 0]] = self.EMPTY

    def get_observations(self):
        """
        Egocentric sensor pipeline reading from batched maps.
        Returns tensor of shape (BATCH_SIZE, NUM_SENSORS)
        """
        if self.num_bugs == 0:
            raise ValueError("World has no population. Call populate() first.")
        
        relative_coords = self.rotated_offsets[self.headings]
        absolute_coords = relative_coords + self.positions.unsqueeze(1)

        x_coords = torch.clamp(absolute_coords[..., 0], 0, self.grid_size - 1)
        y_coords = torch.clamp(absolute_coords[..., 1], 0, self.grid_size - 1)
        
        # We need to tell the GPU which map belongs to which bug.
        # batch_idx shape: (BATCH_SIZE, 1). This broadcasts across the NUM_SENSORS dimension.
        batch_idx = torch.arange(self.num_bugs, device=self.device).unsqueeze(1)
        
        # Crop raw data from the private maps
        raw_views = self.map[batch_idx, y_coords, x_coords]
        # Look up the actual map data for the blocking tiles
        blocked_views = raw_views[:, self.los_blockers]
        # If ANY of the blocking tiles equal WALL, this tile
        # is blocked
        is_blocked = (blocked_views == self.WALL).any(dim=2)
        
        visible_views = torch.where(
            is_blocked,
            torch.tensor(self.EMPTY, dtype=torch.float32, device=self.device),
            raw_views
        )

        # Mask out invisible tiles
        final_vision = visible_views * self.masks

        # Normalize life force to [0, 1]
        normalized_life = self.life_force / self.MAX_LIFE

        if self._obs_buffer is None:
            self._obs_buffer = torch.empty(self.num_bugs, self.num_sensors + 1, device=self.device)

        self._obs_buffer[:, :-1] = final_vision
        self._obs_buffer[:, -1]  = normalized_life

        return self._obs_buffer
    
    def step(self, actions):
        """
        Update the world state based on batched actions from the NN.
        Actions: 0 = Move Forward, 1 = Turn Right, 2 = Turn Left
        """
        # Heading Updates
        turn_right = (actions == 1)
        turn_left = (actions == 2)

        self.headings = torch.where(turn_right, (self.headings + 1) % 4, self.headings)
        self.headings = torch.where(turn_left, (self.headings - 1) % 4, self.headings)

        # Movement Updates
        move_forward = (actions == 0)

        forward_vectors = torch.tensor([[0, -1], [1, 0], [0, 1], [-1, 0]], device=self.device)
        deltas = forward_vectors[self.headings]

        new_positions = self.positions + deltas

        x_check = torch.clamp(new_positions[:, 0], 0, self.grid_size - 1)
        y_check = torch.clamp(new_positions[:, 1], 0, self.grid_size - 1)
        
        # Check collisions against each bug's private map
        batch_idx = torch.arange(self.num_bugs, device=self.device)
        target_tiles = self.map[batch_idx, y_check, x_check]

        valid_move = move_forward & (target_tiles != self.WALL)

        # Survival Logic
        # 1. Did they eat food this frame?
        ate_food = valid_move & (target_tiles == self.FOOD)

        # 2. Update Life Force
        self.life_force -= self.LIFE_DECAY              # Cost of living
        self.life_force += ate_food * self.FOOD_REWARD  # Reward for eating
        self.life_force = torch.clamp(self.life_force, 0, self.MAX_LIFE) # Cap at max

        # 3. Handle Food Consumption & Respawning
        if ate_food.any():
            # Erase the eaten food from the maps
            self.map[batch_idx[ate_food], y_check[ate_food], x_check[ate_food]] = self.EMPTY
            # Spawn new food only in the maps where food was eaten
            self._spawn_food(ate_food)

        self.positions[:, 0] = torch.where(valid_move, new_positions[:, 0], self.positions[:, 0])
        self.positions[:, 1] = torch.where(valid_move, new_positions[:, 1], self.positions[:, 1])

        return self.get_observations()
    

class FunctionalWorld:
    EMPTY = 0
    WALL = 2
    FOOD = 4

    @staticmethod
    def get_single_observation(pos, heading, life, bug_map, rotated_offsets, los_blockers, mask, max_life):
        """
        Pure function: 1 Bug, 1 Map -> 1 Observation Vector
        Uses pre-computed rotated offsets and line-of-sight blocker indices.
        """
        # 1. Get relative sensor offsets based on current heading: shape (num_sensors, 2)
        rel_coords = rotated_offsets[heading]
        abs_coords = rel_coords + pos
        
        # 2. Clamp coordinates to map boundaries
        grid_size = bug_map.shape[0]
        x_coords = torch.clamp(abs_coords[:, 0], 0, grid_size - 1)
        y_coords = torch.clamp(abs_coords[:, 1], 0, grid_size - 1)
        
        # 3. Read the raw tiles from the map: shape (num_sensors)
        raw_views = bug_map[y_coords, x_coords]
        
        # 4. Occlusion logic using pre-computed blockers
        # los_blockers shape: (num_sensors, max_blockers)
        # Look up the tile values at the blocker indices
        blocked_views = raw_views[los_blockers]
        
        # If any tile in the line-of-sight path is a WALL, the sensor is blocked
        is_blocked = (blocked_views == FunctionalWorld.WALL).any(dim=1)
        
        visible_views = torch.where(
            is_blocked,
            torch.tensor(FunctionalWorld.EMPTY, device=bug_map.device, dtype=raw_views.dtype),
            raw_views
        )
        
        # 5. Apply the bug's specific vision mask (for blind spots, etc.)
        final_vision = visible_views * mask
        
        # 6. Append normalized life force
        normalized_life = (life / max_life).unsqueeze(0)
        
        return torch.cat([final_vision, normalized_life], dim=0)

    @staticmethod
    def single_step(action, pos, heading, life, bug_map, forward_vectors, life_decay, food_reward, max_life):
        """
        Pure function: Takes current state, returns NEXT state.
        """
        # 0. ALIVE CHECK: Is this bug currently breathing?
        is_alive = (life > 0.0)

        # 1. Turn updates (Dead bugs can technically still "think" and turn in place, 
        # but they won't move. You can mask this too if you want, but it doesn't affect the map.)
        new_heading = torch.where(action == 1, (heading + 1) % 4, heading)
        new_heading = torch.where(action == 2, (new_heading - 1) % 4, new_heading)
        
        # 2. Movement updates
        move_forward = (action == 0)
        delta = forward_vectors[new_heading]
        attempted_pos = pos + delta
        
        grid_size = bug_map.shape[0]
        x_check = torch.clamp(attempted_pos[0], 0, grid_size - 1)
        y_check = torch.clamp(attempted_pos[1], 0, grid_size - 1)
        
        target_tile = bug_map[y_check, x_check]
        
        # GHOST MASK: You must be alive to have a valid move or eat food
        valid_move = is_alive & move_forward & (target_tile != FunctionalWorld.WALL)
        ate_food = is_alive & valid_move & (target_tile == FunctionalWorld.FOOD)
        
        # 3. Apply position updates
        new_pos = torch.where(valid_move, attempted_pos, pos)
        
        # 4. Apply Life updates
        # If dead, keep life at 0 (prevent zombies from getting food or going negative).
        # If alive, decay life and add food reward.
        new_life = torch.where(
            is_alive,
            torch.clamp(life - life_decay + (ate_food * food_reward), 0.0, max_life),
            torch.tensor(0.0, device=bug_map.device, dtype=life.dtype)
        )
        
        # 5. Map Updates
        new_map = bug_map.clone()
        current_tile = new_map[y_check, x_check]
        
        new_map[y_check, x_check] = torch.where(
            ate_food, # ate_food already requires is_alive to be True
            torch.tensor(FunctionalWorld.EMPTY, device=bug_map.device, dtype=current_tile.dtype),
            current_tile
        )
             
        return new_pos, new_heading, new_life, new_map, ate_food

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    def visualize_state(world, bug_index=0, filename="debug_frame.png"):
        global_map = world.map[bug_index].cpu().numpy()
        pos_x, pos_y = world.positions[bug_index].cpu().numpy()
        heading = world.headings[bug_index].item()

        obs = world.get_observations()
        vision_1d = obs[bug_index, :-1].cpu()  # strip life force

        # Build a 2D egocentric grid to visualize by scattering cone values back
        # into a square grid for display purposes
        R = world.sensor_radius
        grid_dim = 2 * R + 1
        egocentric_view = torch.zeros(grid_dim, grid_dim)

        offsets = world.cone_offsets.cpu()  # (num_sensors, 2) — [x, y] offsets
        for i, (cx, cy) in enumerate(offsets.tolist()):
            # Shift from [-R, R] to [0, grid_dim]
            gx = int(cx) + R
            gy = int(cy) + R
            egocentric_view[gy, gx] = vision_1d[i]

        egocentric_view = egocentric_view.numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        cmap = plt.cm.colors.ListedColormap(['black', 'white', 'white', 'white', 'green'])
        bounds = [0, 1, 3, 5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        ax1.set_title(f"Global Map (Bug {bug_index})")
        ax1.imshow(global_map, cmap=cmap, norm=norm, origin='upper')
        ax1.plot(pos_x, pos_y, 'ro', markersize=8)
        dx, dy = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[heading]
        ax1.arrow(pos_x, pos_y, dx, dy, color='red', head_width=0.5, head_length=0.5)

        ax2.set_title(f"Cone View ({world.num_sensors} sensors)")
        ax2.imshow(egocentric_view, cmap=cmap, norm=norm, origin='upper')
        # Bug is at center of the square display grid
        ax2.plot(R, R, 'ro', markersize=8)
        # Draw heading arrow — always points up (North) since view is egocentric
        ax2.arrow(R, R, 0, -1, color='red', head_width=0.3, head_length=0.3)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        print(f"--> Saved visualization to {filename}")
    # --- SANITY CHECK SCRIPT ---

    # 1. Setup World (Small grid, small sensor radius for easy viewing)
    world = World(grid_size=10, sensor_radius=2, fov_degrees=120, front_fov_radius=2, side_fov_radius=1, device="cuda")

    full_vision = torch.ones(world.num_sensors, dtype=torch.float32, device="cuda")

    world.populate([full_vision])

    # 2. Override with a deterministic layout
    # Clear the map completely
    world.map[0] = world.EMPTY

    # Draw a specific wall pattern (a corner)
    # world.map[0, 2:8, 2] = world.WALL  # Vertical wall
    # world.map[0, 2, 2:8] = world.WALL  # Horizontal wall

    # # Place food in a specific spot
    # world.map[0, 4, 4] = world.FOOD

    # --- THE OCCLUSION TEST ---
    # The bug is at x=5, y=5 and facing NORTH (Up).

    # Place a WALL at Forward 1 (y=4, x=5)
    # world.map[0, 4, 5] = world.WALL
    # # Place a WALL to create a corder at (y=5, x = 6)
    # world.map[0, 5, 6] = world.WALL

    # # Place FOOD at Forward 2 (y=4, x=5)
    # world.map[0, 4, 6] = world.FOOD

    # # 3. Force the bug's position and heading
    # world.positions[0] = torch.tensor([5, 5], device="cuda") # x=5, y=5
    # world.headings[0] = torch.tensor(world.NORTH, device="cuda") # Facing North (Up)

    # print("--- INITIAL STATE (Facing North) ---")
    # visualize_state(world, bug_index=0, filename="step1_north.png")

    # print("--- ACTION: TURN RIGHT ---")
    # actions = torch.tensor([1], device="cuda") # 1 = Turn Right
    # world.step(actions)
    # visualize_state(world, bug_index=0, filename="step2_east.png")

    # print("--- ACTION: MOVE FORWARD ---")
    # actions = torch.tensor([0], device="cuda") # 0 = Move Forward
    # world.step(actions)
    # visualize_state(world, bug_index=0, filename="step3_moved.png")

    # print("--- ACTION: MOVE FORWARD AGAIN (BLOCKED)---")
    # actions = torch.tensor([0], device="cuda") # 0 = Move Forward
    # world.step(actions)
    # visualize_state(world, bug_index=0, filename="step4_moved_blocked.png")
    # 2. Override with a deterministic layout
    world.map[0, 4, 5] = world.WALL  # Wall Forward 1 (North)
    world.map[0, 5, 6] = world.WALL  # Wall Right 1 (East)
    world.map[0, 4, 6] = world.FOOD  # Unreachable Food (Northeast)

    # --- NEW: Place a reachable piece of food directly BEHIND the bug ---
    world.map[0, 6, 5] = world.FOOD

    # 3. Force the bug's position and heading
    world.positions[0] = torch.tensor([5, 5], device="cuda") # x=5, y=5
    world.headings[0] = torch.tensor(world.NORTH, device="cuda") # Facing North (Up)

    print("--- INITIAL STATE (Facing North) ---")
    visualize_state(world, bug_index=0, filename="step1_north.png")

    print("--- ACTION: TURN RIGHT ---")
    actions = torch.tensor([1], device="cuda") # 1 = Turn Right
    world.step(actions)
    visualize_state(world, bug_index=0, filename="step2_east.png")

    print("--- ACTION: MOVE FORWARD (BLOCKED BY WALL) ---")
    actions = torch.tensor([0], device="cuda") # 0 = Move Forward
    world.step(actions)
    visualize_state(world, bug_index=0, filename="step3_blocked.png")

    print("--- ACTION: TURN RIGHT AGAIN (Facing South) ---")
    actions = torch.tensor([1], device="cuda") # 1 = Turn Right
    world.step(actions)
    visualize_state(world, bug_index=0, filename="step4_south.png")

    # --- THE EATING TEST ---
    
    # Artificially lower health so we can see the addition work
    world.life_force[0] = 50.0 
    print(f"\n[Stats] Life Force BEFORE moving: {world.life_force[0].item()}")

    print("--- ACTION: MOVE FORWARD (Eating Food!) ---")
    actions = torch.tensor([0], device="cuda") # 0 = Move Forward
    world.step(actions)
    
    print(f"[Stats] Life Force AFTER moving: {world.life_force[0].item()}\n")
    visualize_state(world, bug_index=0, filename="step5_ate_food.png")