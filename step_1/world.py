import torch


class World:
    """
    PPO world. Drop-in replacement for the GA World.

    What changed from the GA version:
      - No walls. Boundary is a hard clamp, not a wall tile.
      - Food zones instead of random wall maps. Food spawns inside
        named rectangles only.
      - num_envs fixed at construction (not set in populate).
      - reset(env_ids) replaces populate() — resets a subset of envs.
      - step() returns (obs, rewards, dones) instead of just obs.
      - Reward is delta life force — purely what the bug "feels".
      - Vision mask is all-ones (full cone). No per-bug masks needed.

    Everything else — cone offsets, rotation table, LOS blockers,
    get_observations, the step mechanics — is copied verbatim from
    the original GA World which was verified working.
    """

    EMPTY = 0
    FOOD  = 4
    # No WALL constant. The grid boundary is enforced by clamping,
    # not by placing wall tiles. LOS blockers check == WALL which
    # will never fire, so occlusion is effectively disabled (correct
    # for an open arena — re-enable if you add obstacle zones later).

    NORTH = 0
    EAST  = 1
    SOUTH = 2
    WEST  = 3

    def __init__(
        self,
        num_envs,
        grid_size        = 32,
        food_zones       = None,   # list of {"x","y","w","h"} dicts, or None = full grid
        min_food         = 8,
        sensor_radius    = 5,
        fov_degrees      = 120,
        front_fov_radius = 5,
        side_fov_radius  = 2,
        max_life         = 200.0,
        food_reward      = 40.0,
        life_decay       = 1.0,
        device           = "cuda",
    ):
        self.device           = device
        self.num_envs         = num_envs
        self.grid_size        = grid_size
        self.min_food         = min_food
        self.MAX_LIFE         = max_life
        self.FOOD_REWARD      = food_reward
        self.LIFE_DECAY       = life_decay
        self.sensor_radius    = sensor_radius

        # ── Sensor geometry (verbatim from GA World) ──────────────────────────
        self.cone_offsets  = self._compute_cone_offsets(
            fov_degrees, front_fov_radius, side_fov_radius
        )
        self.num_sensors   = len(self.cone_offsets)

        offsets_N = self.cone_offsets
        offsets_E = torch.stack([-self.cone_offsets[:, 1],  self.cone_offsets[:, 0]], dim=1)
        offsets_S = torch.stack([-self.cone_offsets[:, 0], -self.cone_offsets[:, 1]], dim=1)
        offsets_W = torch.stack([ self.cone_offsets[:, 1], -self.cone_offsets[:, 0]], dim=1)
        self.rotated_offsets = torch.stack([offsets_N, offsets_E, offsets_S, offsets_W])

        # ── LOS blockers (verbatim from GA World) ─────────────────────────────
        self.los_blockers = self._compute_line_of_sight_blockers()

        # ── Vision mask — all ones (every bug sees the full cone) ─────────────
        # GA version had per-bug masks for genetic variety. PPO has one agent
        # so we just use a full mask and skip the multiplication overhead.
        self.mask = torch.ones(self.num_sensors, device=device)

        # ── Food zone coords ──────────────────────────────────────────────────
        self.zone_mask   = self._build_zone_mask(food_zones)
        yx               = self.zone_mask.nonzero(as_tuple=False)  # (N,2) [y,x]
        self.zone_coords = torch.stack([yx[:, 1], yx[:, 0]], dim=1)  # → [x,y]

        # Spawn coords: zone interior only, at least 3 from any edge so the
        # bug doesn't immediately face the boundary on spawn.
        interior = (
            (self.zone_coords[:, 0] >= 3) &
            (self.zone_coords[:, 0] <= grid_size - 4) &
            (self.zone_coords[:, 1] >= 3) &
            (self.zone_coords[:, 1] <= grid_size - 4)
        )
        self.spawn_coords = self.zone_coords[interior]
        assert self.spawn_coords.shape[0] > 0, \
            "No valid spawn coords — zone is too small or too close to the edge."

        # ── State tensors ─────────────────────────────────────────────────────
        self.map        = torch.zeros(num_envs, grid_size, grid_size,
                                      dtype=torch.float32, device=device)
        self.positions  = torch.zeros(num_envs, 2, dtype=torch.long,  device=device)
        self.headings   = torch.zeros(num_envs,    dtype=torch.long,  device=device)
        self.life_force = torch.zeros(num_envs,    dtype=torch.float32, device=device)
        self._obs_buffer = None

        # Full reset on construction
        self.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, env_ids=None):
        """
        Reset a subset of envs (or all if env_ids is None).
        Returns full obs tensor (num_envs, obs_dim).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = len(env_ids)

        # Random headings
        self.headings[env_ids] = torch.randint(0, 4, (n,), device=self.device)

        # Start at half life
        self.life_force[env_ids] = self.MAX_LIFE / 2.0

        # Clear maps, seed food, THEN place bug inside the zone.
        # Order matters: food must exist before the bug is positioned
        # so _spawn_food's "not on bug" check works correctly.
        self.map[env_ids] = self.EMPTY

        # Place bug FIRST
        rand_idx = torch.randint(0, self.spawn_coords.shape[0], (n,), device=self.device)
        self.positions[env_ids] = self.spawn_coords[rand_idx]

        # THEN spawn food — not_on_bug check now works correctly
        for _ in range(self.min_food):
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = True
            self._spawn_food(mask)

        self._obs_buffer = None
        return self.get_observations()

    def step(self, actions):
        """
        Actions: 0 = move forward, 1 = turn right, 2 = turn left
        Returns: obs (num_envs, obs_dim), rewards (num_envs,), dones (num_envs,)
        """
        # ── Turns (verbatim from GA World) ────────────────────────────────────
        turn_right = (actions == 1)
        turn_left  = (actions == 2)
        self.headings = torch.where(turn_right, (self.headings + 1) % 4, self.headings)
        self.headings = torch.where(turn_left,  (self.headings - 1) % 4, self.headings)

        # ── Movement ──────────────────────────────────────────────────────────
        move_forward = (actions == 0)
        forward_vectors = torch.tensor(
            [[0, -1], [1, 0], [0, 1], [-1, 0]], device=self.device
        )
        deltas       = forward_vectors[self.headings]
        new_positions = self.positions + deltas

        # Hard boundary clamp (no wall tiles — just stop at edge)
        x_check = torch.clamp(new_positions[:, 0], 0, self.grid_size - 1)
        y_check = torch.clamp(new_positions[:, 1], 0, self.grid_size - 1)

        batch_idx    = torch.arange(self.num_envs, device=self.device)
        target_tiles = self.map[batch_idx, y_check, x_check]

        # No walls to collide with — any forward action moves
        valid_move = move_forward
        ate_food   = valid_move & (target_tiles == self.FOOD)

        # ── Life force & reward ───────────────────────────────────────────────
        self.life_force -= self.LIFE_DECAY
        self.life_force += ate_food.float() * self.FOOD_REWARD
        self.life_force.clamp_(0.0, self.MAX_LIFE)

        # Reward +1.0 for every step lived.
        rewards = torch.ones_like(self.life_force)

        # ── Food respawn (verbatim from GA World) ─────────────────────────────
        if ate_food.any():
            self.map[batch_idx[ate_food], y_check[ate_food], x_check[ate_food]] = self.EMPTY
            self._spawn_food(ate_food)

        # ── Apply position update ─────────────────────────────────────────────
        self.positions[:, 0] = torch.where(valid_move, x_check, self.positions[:, 0])
        self.positions[:, 1] = torch.where(valid_move, y_check, self.positions[:, 1])

        # ── Dones ─────────────────────────────────────────────────────────────
        dones = self.life_force <= 0.0

        return self.get_observations(), rewards, dones

    def get_observations(self):
        """
        Verbatim from GA World.get_observations(), minus the per-bug mask.
        Returns (num_envs, num_sensors + 1).
        """
        relative_coords = self.rotated_offsets[self.headings]
        absolute_coords = relative_coords + self.positions.unsqueeze(1)

        x_coords = torch.clamp(absolute_coords[..., 0], 0, self.grid_size - 1)
        y_coords = torch.clamp(absolute_coords[..., 1], 0, self.grid_size - 1)

        batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        raw_views = self.map[batch_idx, y_coords, x_coords]

        # LOS occlusion — checks for WALL which never exists in this world,
        # so is_blocked is always False. Kept for forward-compatibility.
        blocked_views = raw_views[:, self.los_blockers]
        is_blocked    = (blocked_views == 2).any(dim=2)  # 2 = WALL (never placed)

        visible_views = torch.where(
            is_blocked,
            torch.tensor(self.EMPTY, dtype=torch.float32, device=self.device),
            raw_views,
        )

        # Full vision mask (all ones — no blind spots)
        final_vision = visible_views * self.mask

        normalized_life = self.life_force / self.MAX_LIFE

        if self._obs_buffer is None:
            self._obs_buffer = torch.empty(
                self.num_envs, self.num_sensors + 1, device=self.device
            )

        self._obs_buffer[:, :-1] = final_vision
        self._obs_buffer[:, -1]  = normalized_life

        return self._obs_buffer

    @property
    def obs_dim(self):
        return self.num_sensors + 1

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_zone_mask(self, food_zones):
        mask = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool, device=self.device)
        if food_zones is None:
            mask[1:-1, 1:-1] = True
            return mask
        for z in food_zones:
            x1 = max(z["x"], 1)
            y1 = max(z["y"], 1)
            x2 = min(z["x"] + z["w"], self.grid_size - 1)
            y2 = min(z["y"] + z["h"], self.grid_size - 1)
            mask[y1:y2, x1:x2] = True
        return mask

    def _spawn_food(self, needs_food_mask):
        """
        Spawn one food token per env in needs_food_mask, inside zone_coords only.
        Same retry logic as GA World._spawn_food.
        """
        active    = needs_food_mask.clone()
        n_zone    = self.zone_coords.shape[0]
        batch_idx = torch.arange(self.num_envs, device=self.device)

        for _ in range(5):
            rand_idx = torch.randint(0, n_zone, (self.num_envs, 10), device=self.device)
            rx = self.zone_coords[rand_idx, 0]  # (num_envs, 10)
            ry = self.zone_coords[rand_idx, 1]

            for i in range(10):
                is_empty   = (self.map[batch_idx, ry[:, i], rx[:, i]] == self.EMPTY)
                not_on_bug = ~(
                    (rx[:, i] == self.positions[:, 0]) &
                    (ry[:, i] == self.positions[:, 1])
                )
                valid = active & is_empty & not_on_bug
                self.map[batch_idx[valid], ry[valid, i], rx[valid, i]] = self.FOOD
                active &= ~valid

            if not active.any():
                break

    # ── Sensor geometry (verbatim from GA World) ──────────────────────────────

    def _compute_cone_offsets(self, fov_degrees=120, front_radius=5, side_radius=2):
        max_radius = max(front_radius, side_radius)
        y, x = torch.meshgrid(
            torch.arange(-max_radius, max_radius + 1, device=self.device),
            torch.arange(-max_radius, max_radius + 1, device=self.device),
            indexing='ij',
        )
        offsets     = torch.stack([x.flatten(), y.flatten()], dim=1).float()
        angles      = torch.atan2(offsets[:, 0], -offsets[:, 1])
        half_fov    = torch.tensor(fov_degrees / 2 * (3.14159 / 180), device=self.device)
        angle_ratio = (angles.abs() / half_fov).clamp(0, 1)
        eff_radius  = front_radius + (side_radius - front_radius) * angle_ratio
        distances   = torch.sqrt(offsets[:, 0] ** 2 + offsets[:, 1] ** 2)
        cone_mask   = (
            (distances <= eff_radius) &
            (angles.abs() <= half_fov) &
            ((offsets[:, 0] != 0) | (offsets[:, 1] != 0))
        )
        return offsets[cone_mask].long()

    def _compute_line_of_sight_blockers(self):
        R            = self.sensor_radius
        grid_dim     = 2 * R + 1
        center       = R
        max_blockers = R * 4

        flat_to_cone = torch.full((grid_dim * grid_dim,), -1, dtype=torch.long)
        for cone_i, (cx, cy) in enumerate(self.cone_offsets.cpu().tolist()):
            tx = int(cx) + center
            ty = int(cy) + center
            flat_to_cone[ty * grid_dim + tx] = cone_i

        pad_idx         = 0
        blocker_indices = torch.full(
            (self.num_sensors, max_blockers), pad_idx, dtype=torch.long
        )

        def bresenham(x0, y0, x1, y1):
            points = []
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            x, y   = x0, y0
            sx     = 1 if x0 < x1 else -1
            sy     = 1 if y0 < y1 else -1
            err    = dx - dy
            while True:
                points.append((x, y))
                if x == x1 and y == y1:
                    break
                prev_x, prev_y = x, y
                e2 = 2 * err
                moved_x = moved_y = False
                if e2 > -dy:
                    err -= dy; x += sx; moved_x = True
                if e2 <  dx:
                    err += dx; y += sy; moved_y = True
                if moved_x and moved_y:
                    points.append((prev_x, y))
                    points.append((x, prev_y))
            return points

        for i, (cx, cy) in enumerate(self.cone_offsets.cpu().tolist()):
            tx   = int(cx) + center
            ty   = int(cy) + center
            path = bresenham(center, center, tx, ty)
            for step, (bx, by) in enumerate(path[1:-1]):
                cone_idx = flat_to_cone[by * grid_dim + bx].item()
                if cone_idx >= 0:
                    blocker_indices[i, step] = cone_idx

        return blocker_indices.to(self.device)