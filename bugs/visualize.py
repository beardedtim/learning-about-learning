"""
Bug Simulation Visualizer
Usage: python visualize.py [path_to_checkpoint.pt]
       python visualize.py  (will prompt for path)

Controls:
  SPACE       - Pause / Resume
  R           - Restart simulation
  Left/Right  - Adjust speed (also use the slider)
  Q / ESC     - Quit
"""

import sys
import os
import math
import argparse
import torch
import pygame

# ── bring in your existing modules ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from world import World, FunctionalWorld
from species import GeneticColonySpecies
from checkpoint import load_champion

# ── Layout constants ───────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1280, 800
WORLD_W  = 700          # left panel width
PANEL_W  = SCREEN_W - WORLD_W  # right panel width
PADDING  = 16

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG           = (18,  18,  24)
C_PANEL        = (28,  28,  38)
C_BORDER       = (60,  60,  80)
C_EMPTY        = (30,  32,  44)
C_WALL         = (80,  85, 110)
C_FOOD         = (80, 220,  80)
C_BUG          = (255, 200,  40)
C_BUG_OUTLINE  = (255, 255, 200)
C_CONE_EMPTY   = (60,  80, 120, 60)    # RGBA
C_CONE_FOOD    = (80, 220,  80, 90)
C_CONE_WALL    = (200,  80,  80, 90)
C_HEADING_ARROW= (255, 120,  40)
C_TEXT_HEAD    = (200, 210, 255)
C_TEXT_BODY    = (150, 160, 200)
C_TEXT_DIM     = (90,  95, 130)
C_HEALTH_HI    = (80, 220,  80)
C_HEALTH_LO    = (220,  80,  80)
C_HEALTH_BG    = (40,  40,  60)
C_ACCENT       = (100, 140, 255)
C_BTN          = (50,  55,  80)
C_BTN_HOV      = (70,  80, 120)
C_BTN_ACT      = (100, 130, 220)
C_SLIDER_TRACK = (50,  55,  80)
C_SLIDER_FILL  = (100, 140, 255)
C_SLIDER_KNOB  = (200, 210, 255)
C_ACTION_0     = (100, 200, 255)  # Move Forward
C_ACTION_1     = (255, 180,  60)  # Turn Right
C_ACTION_2     = (180, 130, 255)  # Turn Left

ACTION_NAMES   = ["Move Forward", "Turn Right", "Turn Left"]
ACTION_COLORS  = [C_ACTION_0, C_ACTION_1, C_ACTION_2]
HEADING_NAMES  = ["North ↑", "East →", "South ↓", "West ←"]

# Speed settings: (label, steps_per_frame, delay_ms)
SPEED_LEVELS = [
    ("1×",   1, 120),
    ("2×",   1,  60),
    ("4×",   2,  30),
    ("8×",   4,  16),
    ("16×",  8,   8),
    ("32×", 16,   0),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i]-a[i])*t) for i in range(3))

def rounded_rect(surf, color, rect, r=8, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=r)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=r)

def text(surf, msg, font, color, x, y, anchor="topleft"):
    s = font.render(str(msg), True, color)
    r = s.get_rect(**{anchor: (x, y)})
    surf.blit(s, r)
    return r

def bar(surf, rect, value, max_val, lo_color=C_HEALTH_LO, hi_color=C_HEALTH_HI, bg=C_HEALTH_BG, radius=4):
    t = value / max(max_val, 1)
    pygame.draw.rect(surf, bg, rect, border_radius=radius)
    fill = rect.inflate(0, 0)
    fill.width = max(4, int(rect.width * t))
    pygame.draw.rect(surf, lerp_color(lo_color, hi_color, t), fill, border_radius=radius)

# ── Simulation state ───────────────────────────────────────────────────────────
class SimState:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.forward_vectors = torch.tensor([[0, -1], [1, 0], [0, 1], [-1, 0]], device=self.device)
        self._load()

    def _load(self):
        print(f"[Viz] Loading champion from {self.checkpoint_path} ...")
        
        # The new brain is a standard stateless PyTorch module. load_champion hydrates it!
        self.brain, data = load_champion(GeneticColonySpecies, self.checkpoint_path, device=self.device)
        self.brain.eval() # Set to evaluation mode
        
        cfg = data["world_config"]
        print(f"Config: {cfg}")
        self.world = World(
            grid_size       = cfg["grid_size"],
            sensor_radius   = cfg["sensor_radius"],
            fov_degrees     = cfg["fov_degrees"],
            front_fov_radius= cfg["front_fov_radius"],
            side_fov_radius = cfg["side_fov_radius"],
            max_life        = cfg["max_life"],
            food_reward     = cfg["food_reward"],
            life_decay      = cfg["life_decay"],
            min_food        = cfg["min_food"],
            device          = self.device,
        )

        num_sensors = self.world.num_sensors
        vision_mask = [torch.ones(num_sensors, device=self.device)]
        self.world.populate(vision_mask, wall_density=0.05)

        # Initialize the stateless memory for a single bug
        self.memory = torch.zeros(self.brain.hidden_dim, device=self.device)

        # Get the first observation using the FunctionalWorld
        self.obs = FunctionalWorld.get_single_observation(
            self.world.positions[0],
            self.world.headings[0],
            self.world.life_force[0],
            self.world.map[0],
            self.world.rotated_offsets,
            self.world.los_blockers,
            self.world.masks[0],
            self.world.MAX_LIFE
        )

        self.step_num = 0
        self.food_eaten = 0
        self.last_action = 0
        self.last_life   = self.world.life_force[0].item()
        self.alive       = True
        print(f"[Viz] World {cfg['grid_size']}×{cfg['grid_size']}, "
              f"{num_sensors} sensors, max_life={cfg['max_life']}")

    def restart(self):
        self._load()

    def step(self):
        if not self.alive:
            return
            
        # 1. Brain thinks (Stateless)
        action, self.memory = self.brain(self.obs, self.memory)
        self.last_action = action.item()
        
        # 2. Physics step
        new_pos, new_heading, new_life, new_map, ate_food = FunctionalWorld.single_step(
            action,
            self.world.positions[0],
            self.world.headings[0],
            self.world.life_force[0],
            self.world.map[0],
            self.forward_vectors,
            self.world.LIFE_DECAY,
            self.world.FOOD_REWARD,
            self.world.MAX_LIFE
        )
        
        # 3. Map state back to the visualizer's world instance so the UI can draw it
        self.world.positions[0] = new_pos
        self.world.headings[0] = new_heading
        self.world.life_force[0] = new_life
        self.world.map[0] = new_map
        
        # 4. Camera gets new frame
        self.obs = FunctionalWorld.get_single_observation(
            new_pos, new_heading, new_life, new_map,
            self.world.rotated_offsets,
            self.world.los_blockers,
            self.world.masks[0],
            self.world.MAX_LIFE
        )
        
        # Update stats and handle food respawning
        if ate_food:
            self.food_eaten += 1
            # Respawning food so the visualizer doesn't eventually run out
            needs_food_mask = torch.tensor([True], device=self.device)
            self.world._spawn_food(needs_food_mask)
            
            # Recalculate observation just in case food spawned directly in front of the bug!
            self.obs = FunctionalWorld.get_single_observation(
                new_pos, new_heading, new_life, self.world.map[0],
                self.world.rotated_offsets,
                self.world.los_blockers,
                self.world.masks[0],
                self.world.MAX_LIFE
            )

        self.step_num += 1
        self.last_life = new_life.item()
        if self.last_life <= 0:
            self.alive = False

    # ── query helpers ──────────────────────────────────────────────────────────
    @property
    def grid_size(self): return self.world.grid_size
    @property
    def pos(self): return self.world.positions[0].cpu().tolist()      # [x, y]
    @property
    def heading(self): return self.world.headings[0].item()
    @property
    def life(self): return self.world.life_force[0].item()
    @property
    def max_life(self): return self.world.MAX_LIFE
    @property
    def map_np(self): return self.world.map[0].cpu()                  # tensor HxW
    
    # Updated: Since this is a single unbatched pass, obs is a 1D tensor
    @property
    def vision_1d(self): return self.obs[:-1].cpu()                   
    
    @property
    def cone_offsets(self): return self.world.cone_offsets.cpu()      # (N, 2)  x,y

# ── Slider widget ──────────────────────────────────────────────────────────────
class Slider:
    def __init__(self, rect, levels, start=0):
        self.rect    = pygame.Rect(rect)
        self.levels  = levels
        self.index   = start
        self.dragging= False

    @property
    def n(self): return len(self.levels)

    def _knob_x(self):
        if self.n == 1: return self.rect.centerx
        frac = self.index / (self.n - 1)
        return int(self.rect.left + frac * self.rect.width)

    def draw(self, surf, font_sm):
        # track
        track = pygame.Rect(self.rect.left, self.rect.centery - 3, self.rect.width, 6)
        pygame.draw.rect(surf, C_SLIDER_TRACK, track, border_radius=3)
        if self.n > 1:
            frac = self.index / (self.n - 1)
            fill = pygame.Rect(track.left, track.top, int(track.width * frac), track.height)
            pygame.draw.rect(surf, C_SLIDER_FILL, fill, border_radius=3)
        # ticks + labels
        for i, (label, _, _) in enumerate(self.levels):
            frac = i / max(self.n - 1, 1)
            tx = int(self.rect.left + frac * self.rect.width)
            pygame.draw.line(surf, C_BORDER, (tx, self.rect.centery - 6), (tx, self.rect.centery + 6))
            col = C_TEXT_HEAD if i == self.index else C_TEXT_DIM
            text(surf, label, font_sm, col, tx, self.rect.bottom + 2, anchor="midtop")
        # knob
        kx = self._knob_x()
        pygame.draw.circle(surf, C_SLIDER_KNOB, (kx, self.rect.centery), 9)
        pygame.draw.circle(surf, C_ACCENT,      (kx, self.rect.centery), 9, 2)

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            kx = self._knob_x()
            if abs(e.pos[0] - kx) < 14 and abs(e.pos[1] - self.rect.centery) < 14:
                self.dragging = True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging = False
        if e.type == pygame.MOUSEMOTION and self.dragging:
            frac = (e.pos[0] - self.rect.left) / max(self.rect.width, 1)
            self.index = int(round(max(0, min(1, frac)) * (self.n - 1)))

    def bump(self, delta):
        self.index = max(0, min(self.n - 1, self.index + delta))

# ── Button widget ──────────────────────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, key=None):
        self.rect  = pygame.Rect(rect)
        self.label = label
        self.key   = key
        self._hov  = False
        self._act  = False
        self.clicked = False

    def draw(self, surf, font, active=False):
        col = C_BTN_ACT if (active or self._act) else (C_BTN_HOV if self._hov else C_BTN)
        rounded_rect(surf, col, self.rect, r=6)
        text(surf, self.label, font, C_TEXT_HEAD, *self.rect.center, anchor="center")

    def handle_event(self, e):
        self.clicked = False
        if e.type == pygame.MOUSEMOTION:
            self._hov = self.rect.collidepoint(e.pos)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            self.clicked = True
        if e.type == pygame.KEYDOWN and self.key and e.key == self.key:
            self.clicked = True

# ── World renderer ─────────────────────────────────────────────────────────────
def draw_world(surf, sim, cell, ox, oy, cone_surf):
    """Render the grid, cone overlay, and bug onto surf."""
    gs   = sim.grid_size
    grid = sim.map_np
    bx, by = sim.pos
    heading = sim.heading

    # Draw grid cells
    for gy in range(gs):
        for gx in range(gs):
            v = grid[gy, gx].item()
            if v == World.WALL:
                col = C_WALL
            elif v == World.FOOD:
                col = C_FOOD
            else:
                col = C_EMPTY
            r = pygame.Rect(ox + gx*cell, oy + gy*cell, cell-1, cell-1)
            pygame.draw.rect(surf, col, r)

    # -- Cone of vision overlay --
    cone_surf.fill((0, 0, 0, 0))
    offsets  = sim.cone_offsets   # (N, 2) — x, y relative to heading=NORTH
    vision   = sim.vision_1d      # (N,)

    # Rotate offsets to world space depending on heading
    for i, (cx, cy) in enumerate(offsets.tolist()):
        h = heading
        if h == World.NORTH:
            wx, wy = cx, cy
        elif h == World.EAST:
            wx, wy = -cy, cx
        elif h == World.SOUTH:
            wx, wy = -cx, -cy
        else:  # WEST
            wx, wy = cy, -cx
        gx2 = bx + int(wx)
        gy2 = by + int(wy)
        if 0 <= gx2 < gs and 0 <= gy2 < gs:
            v = vision[i].item()
            if v == World.FOOD:
                cone_col = C_CONE_FOOD
            elif v == World.WALL:
                cone_col = C_CONE_WALL
            else:
                cone_col = C_CONE_EMPTY
            cr = pygame.Rect(ox + gx2*cell, oy + gy2*cell, cell-1, cell-1)
            pygame.draw.rect(cone_surf, cone_col, cr)

    surf.blit(cone_surf, (0, 0))

    # -- Bug --
    bug_cx = ox + bx*cell + cell//2
    bug_cy = oy + by*cell + cell//2
    r_bug  = max(3, cell//2 - 1)
    pygame.draw.circle(surf, C_BUG_OUTLINE, (bug_cx, bug_cy), r_bug + 2)
    pygame.draw.circle(surf, C_BUG,         (bug_cx, bug_cy), r_bug)

    # -- Heading arrow --
    dx_map = [0, 1, 0, -1]
    dy_map = [-1, 0, 1, 0]
    dx, dy  = dx_map[heading], dy_map[heading]
    arr_len = cell * 1.4
    ex = bug_cx + dx * arr_len
    ey = bug_cy + dy * arr_len
    pygame.draw.line(surf, C_HEADING_ARROW, (bug_cx, bug_cy), (int(ex), int(ey)), max(2, cell//5))
    # arrowhead
    angle = math.atan2(dy, dx)
    for side in (+0.4, -0.4):
        hx = ex - math.cos(angle + side) * arr_len * 0.35
        hy = ey - math.sin(angle + side) * arr_len * 0.35
        pygame.draw.line(surf, C_HEADING_ARROW, (int(ex), int(ey)), (int(hx), int(hy)), max(2, cell//5))

# ── Egocentric mini-map ────────────────────────────────────────────────────────
def draw_ego_map(surf, sim, rect):
    """Draw the bug's-eye-view sensor grid in `rect`."""
    offsets = sim.cone_offsets   # (N, 2) — relative x, y (North-up)
    vision  = sim.vision_1d

    sr = sim.world.sensor_radius
    grid_dim = 2*sr + 1
    cell  = min(rect.width, rect.height) // grid_dim
    ox = rect.left + (rect.width  - cell*grid_dim)//2
    oy = rect.top  + (rect.height - cell*grid_dim)//2

    # background
    for row in range(grid_dim):
        for col in range(grid_dim):
            r = pygame.Rect(ox + col*cell, oy + row*cell, cell-1, cell-1)
            pygame.draw.rect(surf, C_EMPTY, r)

    # sensor tiles (North-up: +x is right, -y is up)
    for i, (cx, cy) in enumerate(offsets.tolist()):
        gx = int(cx) + sr
        gy = int(cy) + sr
        v  = vision[i].item()
        if v == World.WALL:
            col = C_WALL
        elif v == World.FOOD:
            col = C_FOOD
        else:
            col = (55, 65, 95)      # visible-empty, slightly brighter
        r = pygame.Rect(ox + gx*cell, oy + gy*cell, cell-1, cell-1)
        pygame.draw.rect(surf, col, r)

    # bug dot at center (always facing up in ego view)
    cx_px = ox + sr*cell + cell//2
    cy_px = oy + sr*cell + cell//2
    pygame.draw.circle(surf, C_BUG_OUTLINE, (cx_px, cy_px), max(3, cell//2))
    pygame.draw.circle(surf, C_BUG,         (cx_px, cy_px), max(2, cell//2 - 1))
    # heading arrow (always up)
    arrow_len = cell * 1.2
    pygame.draw.line(surf, C_HEADING_ARROW,
                     (cx_px, cy_px), (cx_px, int(cy_px - arrow_len)), max(1, cell//4))

    # border
    border_rect = pygame.Rect(ox, oy, cell*grid_dim, cell*grid_dim)
    pygame.draw.rect(surf, C_BORDER, border_rect, 1)

# ── Right panel ────────────────────────────────────────────────────────────────
def draw_panel(surf, sim, fonts, btns, slider, panel_x):
    f_lg, f_md, f_sm, f_xs = fonts
    px = panel_x + PADDING
    pw = PANEL_W - PADDING*2
    y  = PADDING

    # ── Title ────────────────────────────────────────────────────────────────
    text(surf, "BUG  VIEWER", f_lg, C_TEXT_HEAD, px, y)
    y += f_lg.get_height() + 4
    text(surf, f"Champion · {sim.grid_size}×{sim.grid_size} world", f_xs, C_TEXT_DIM, px, y)
    y += f_xs.get_height() + PADDING

    pygame.draw.line(surf, C_BORDER, (panel_x, y), (SCREEN_W, y))
    y += PADDING

    # ── Stats ────────────────────────────────────────────────────────────────
    text(surf, "STATS", f_sm, C_ACCENT, px, y)
    y += f_sm.get_height() + 6

    stats = [
        ("Step",       f"{sim.step_num:,}"),
        ("Food eaten", f"{sim.food_eaten}"),
        ("Heading",    HEADING_NAMES[sim.heading]),
        ("Position",   f"({sim.pos[0]}, {sim.pos[1]})"),
    ]
    for label, val in stats:
        text(surf, label, f_sm, C_TEXT_DIM, px, y)
        text(surf, val,   f_sm, C_TEXT_HEAD, px + pw, y, anchor="topright")
        y += f_sm.get_height() + 4

    # Health bar
    y += 4
    text(surf, "Life Force", f_sm, C_TEXT_DIM, px, y)
    lf_str = f"{sim.life:.0f} / {sim.max_life:.0f}"
    text(surf, lf_str, f_sm, C_TEXT_HEAD, px + pw, y, anchor="topright")
    y += f_sm.get_height() + 4
    bar_rect = pygame.Rect(px, y, pw, 14)
    bar(surf, bar_rect, sim.life, sim.max_life)
    y += 20 + PADDING

    if not sim.alive:
        died_surf = f_md.render("✖  DIED", True, (220, 80, 80))
        surf.blit(died_surf, (px, y))
        y += f_md.get_height() + 4

    pygame.draw.line(surf, C_BORDER, (panel_x, y), (SCREEN_W, y))
    y += PADDING

    # ── Last action ──────────────────────────────────────────────────────────
    text(surf, "LAST ACTION", f_sm, C_ACCENT, px, y)
    y += f_sm.get_height() + 8

    for i, (aname, acol) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
        active = (i == sim.last_action)
        bg_col = (*acol[:3], 180) if active else (50, 55, 80)
        r = pygame.Rect(px, y, pw, 28)
        if active:
            pygame.draw.rect(surf, acol, r, border_radius=5)
            text(surf, f"▶  {aname}", f_sm, (20, 20, 20), r.centerx, r.centery, anchor="center")
        else:
            pygame.draw.rect(surf, C_BTN, r, border_radius=5)
            text(surf, aname, f_sm, C_TEXT_DIM, r.centerx, r.centery, anchor="center")
        y += 32

    y += PADDING
    pygame.draw.line(surf, C_BORDER, (panel_x, y), (SCREEN_W, y))
    y += PADDING

    # ── Egocentric view ──────────────────────────────────────────────────────
    text(surf, "BUG'S EYE VIEW", f_sm, C_ACCENT, px, y)
    y += f_sm.get_height() + 6

    ego_size = min(pw, SCREEN_H - y - 180)
    ego_rect = pygame.Rect(px, y, pw, ego_size)
    draw_ego_map(surf, sim, ego_rect)
    y += ego_size + PADDING

    pygame.draw.line(surf, C_BORDER, (panel_x, y), (SCREEN_W, y))
    y += PADDING

    # ── Controls ─────────────────────────────────────────────────────────────
    text(surf, "CONTROLS", f_sm, C_ACCENT, px, y)
    y += f_sm.get_height() + 8

    # Buttons
    bw = (pw - 8) // 2
    for i, btn in enumerate(btns):
        btn.rect.left  = px + i*(bw+8)
        btn.rect.top   = y
        btn.rect.width = bw
        btn.rect.height= 34
        btn.draw(surf, f_sm, active=(btn.label == "Pause" and not sim.alive))

    y += 34 + PADDING + 8

    # Speed slider
    text(surf, "Speed", f_sm, C_TEXT_DIM, px, y)
    y += f_sm.get_height() + 6
    slider.rect = pygame.Rect(px, y, pw, 20)
    slider.draw(surf, f_xs)

# ── Main ───────────────────────────────────────────────────────────────────────
def main(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Viz] Device: {device}")

    sim = SimState(checkpoint_path, device)

    pygame.init()
    pygame.display.set_caption("Bug Visualizer")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()

    # Fonts
    def font(size, bold=False):
        return pygame.font.SysFont("consolas,monospace", size, bold=bold)
    f_lg = font(22, bold=True)
    f_md = font(18, bold=True)
    f_sm = font(14)
    f_xs = font(11)
    fonts = (f_lg, f_md, f_sm, f_xs)

    # Determine cell size so grid fits in left panel with padding
    avail_w = WORLD_W - PADDING*2
    avail_h = SCREEN_H - PADDING*2
    cell = min(avail_w // sim.grid_size, avail_h // sim.grid_size)
    cell = max(4, cell)
    grid_px_w = cell * sim.grid_size
    grid_px_h = cell * sim.grid_size
    ox = PADDING + (avail_w - grid_px_w)//2
    oy = PADDING + (avail_h - grid_px_h)//2

    # Off-screen surface for cone transparency
    cone_surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)

    # Buttons and slider
    btn_pause   = Button(pygame.Rect(0,0,1,1), "Pause",   key=pygame.K_SPACE)
    btn_restart = Button(pygame.Rect(0,0,1,1), "Restart", key=pygame.K_r)
    btns        = [btn_pause, btn_restart]
    speed_slider = Slider((0,0,100,20), SPEED_LEVELS, start=2)

    paused     = False
    step_accum = 0
    last_step_time = pygame.time.get_ticks()

    running = True
    while running:
        dt = clock.tick(60)

        # ── Events ──────────────────────────────────────────────────────────
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_LEFT:
                speed_slider.bump(-1)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_RIGHT:
                speed_slider.bump(+1)

            speed_slider.handle_event(e)
            for btn in btns:
                btn.handle_event(e)

            if btn_pause.clicked:
                paused = not paused
            if btn_restart.clicked:
                sim.restart()
                paused = False
                last_step_time = pygame.time.get_ticks()

        # ── Simulation tick ──────────────────────────────────────────────────
        if not paused and sim.alive:
            label, steps_per_frame, delay_ms = SPEED_LEVELS[speed_slider.index]
            now = pygame.time.get_ticks()
            if (now - last_step_time) >= delay_ms:
                for _ in range(steps_per_frame):
                    sim.step()
                last_step_time = now

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        # Left panel background
        rounded_rect(screen, C_PANEL, pygame.Rect(0, 0, WORLD_W, SCREEN_H), r=0)

        # Grid
        draw_world(screen, sim, cell, ox, oy, cone_surf)

        # Panel title "WORLD" at top of left panel
        text(screen, "WORLD", f_sm, C_TEXT_DIM, PADDING, PADDING//2)

        # Pause / died overlay on world
        if paused:
            ol = pygame.Surface((WORLD_W, SCREEN_H), pygame.SRCALPHA)
            ol.fill((0,0,0,90))
            screen.blit(ol, (0,0))
            text(screen, "PAUSED", f_lg, C_TEXT_HEAD, WORLD_W//2, SCREEN_H//2, anchor="center")
        elif not sim.alive:
            ol = pygame.Surface((WORLD_W, SCREEN_H), pygame.SRCALPHA)
            ol.fill((80,0,0,70))
            screen.blit(ol, (0,0))
            text(screen, "DIED — press R to restart", f_md, (255,120,120), WORLD_W//2, SCREEN_H//2, anchor="center")

        # Divider
        pygame.draw.line(screen, C_BORDER, (WORLD_W, 0), (WORLD_W, SCREEN_H), 2)

        # Right panel
        draw_panel(screen, sim, fonts, btns, speed_slider, WORLD_W + 2)

        btn_pause.label = "Resume" if paused else "Pause"

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bug Simulation Visualizer")
    parser.add_argument("checkpoint", nargs="?", default="bug-fov120-radius5-front5-side2-0.pt",
                        help="Path to .pt checkpoint file (e.g. bug.pt)")
    args = parser.parse_args()

    path = args.checkpoint
    if not path:
        path = input("Enter path to .pt checkpoint file: ").strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)

    main(path)