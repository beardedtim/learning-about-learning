from log import Log
import random

#
# DEFAULTS

# How many times we want to run each Bug through the trial?
NUM_EPOCHS = 1000

# Position is (0, 0) top left, (MAX_X, MAX_Y bottom right)
# Max width we allow for the world
MAX_X = 20
# Max height we allow for the world
MAX_Y = 20

PLAYER_START = (10, 10)

DEFAULT_INITIAL_FOOD_COUNT = 12

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

DEFAULT_VISION_CONE = VISION_CONES.get("Balanced")


def generate_initial_food(num_items=DEFAULT_INITIAL_FOOD_COUNT):
    """
    Randomly generates a list of unique food coordinates.
    Ensures food does not spawn on the player's starting position.
    """
    food_positions = set() # Using a set automatically prevents duplicate coordinates
    
    while len(food_positions) < num_items:
        # Generate random x and y coordinates within the grid bounds
        x = random.randint(0, MAX_X)
        y = random.randint(0, MAX_Y)
        new_pos = (x, y)
        
        # Add to our set only if it isn't the player's starting spot
        if new_pos != PLAYER_START and new_pos not in food_positions:
            food_positions.add(new_pos)
            
    return list(food_positions)

# Max moves we allow between food
LIFE_FORCE = 40
MAX_ITERATIONS = 50000

#
# Genetic Algo Defaults
#
GENERATIONS = 50
POP_SIZE=1000
MUTATION_RATE=0.075

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

class World:
    def __init__(self, 
                 max_x=MAX_X, 
                 max_y=MAX_Y, 
                 player_start=PLAYER_START, 
                 initial_food=generate_initial_food(),
                 facing=NORTH
                 ):
        self.state_dict = {}
        self.player_loc = player_start
        
        self.player_facing = facing 
        
        self.MAX_X = max_x
        self.MAX_Y = max_y

        self.state_dict[self.player_loc] = PLAYER_CHAR

        for food in initial_food:
            self.state_dict[food] = FOOD_CHAR

    def draw_viewport(self, view_radius=5):
        """
        Draws a grid around the player. 
        """
        (p_x, p_y) = self.player_loc
        facing_str = FACING_NAMES.get(self.player_facing, "UNKNOWN")
        
        print(f"--- Absolute World | Centered at ({p_x}, {p_y}) | Facing: {facing_str} ---")
        
        for y in range(p_y - view_radius, p_y + view_radius + 1):
            row_chars = []

            for x in range(p_x - view_radius, p_x + view_radius + 1):
                if x < 0 or x > MAX_X or y < 0 or y > MAX_Y:
                    row_chars.append(WALL_CHAR)
                elif (x, y) == self.player_loc:
                    row_chars.append(PLAYER_ARROWS.get(self.player_facing, PLAYER_CHAR))
                elif (x, y) in self.state_dict:
                    row_chars.append(self.state_dict[(x, y)])
                else:
                    row_chars.append(EMPTY_CHAR)
    
            print(" ".join(row_chars))

    def get_line_of_sight(self, distance, dx, dy):
        p_x, p_y = self.player_loc
        results = []
    
        for step in range(1, distance + 1):
            target_x = p_x + (dx * step)
            target_y = p_y + (dy * step)
            
            if target_x == 0 or target_x == MAX_X or target_y == 0 or target_y == MAX_Y:
                results.append(WALL_CHAR)
            elif not (0 <= target_x <= MAX_X and 0 <= target_y <= MAX_Y):
                results.append(VOID_CHAR)
            else:
                results.append(self.state_dict.get((target_x, target_y), EMPTY_CHAR))
                
        return results
    
    def get_perception(
        self, 
        forward=0, left=0, back=0, right=0, 
        forward_left=0, forward_right=0, back_left=0, back_right=0
    ):
        fx, fy = self.player_facing
        
        # 1. Primary Vectors
        forward_dx, forward_dy = fx, fy
        back_dx, back_dy       = -fx, -fy
        left_dx, left_dy       = fy, -fx
        right_dx, right_dy     = -fy, fx

        # 2. Diagonal Vectors 
        fl_dx, fl_dy = forward_dx + left_dx, forward_dy + left_dy
        fr_dx, fr_dy = forward_dx + right_dx, forward_dy + right_dy
        bl_dx, bl_dy = back_dx + left_dx, back_dy + left_dy
        br_dx, br_dy = back_dx + right_dx, back_dy + right_dy

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
        """Spawns a single piece of food in a random, empty location."""
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
        Moves the player based on a relative action.
        Updates orientation, handles boundaries, and spawns new food when eaten.
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

        # 1c. CLAMP THE VECTOR (Crucial for 8-way grids)
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
                    # Only penalize walls if they are the immediate next step
                    if distance == 1:
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

class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # If no weights are provided, generate a completely random brain
        if weights is None:
            # W1: Weights from Input to Hidden Layer
            self.W1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
            self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
            
            # W2: Weights from Hidden to Output Layer
            self.W2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
            self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]
        else:
            self.W1, self.b1, self.W2, self.b2 = weights

    def forward(self, inputs):
        """Passes the vision data through the network to get movement scores."""
        
        # 1. Hidden Layer Math: Z = W * X + b
        hidden = []
        for i in range(self.hidden_size):
            activation = self.b1[i]
            for j in range(self.input_size):
                activation += inputs[j] * self.W1[i][j]
                
            # ReLU Activation Function (converts negative numbers to 0)
            hidden.append(max(0.0, activation))

        # 2. Output Layer Math
        outputs = []
        for i in range(self.output_size):
            activation = self.b2[i]
            for j in range(self.hidden_size):
                activation += hidden[j] * self.W2[i][j]
            outputs.append(activation) 

        return outputs

    def mutate(self, rate=0.05):
        """Creates a slightly altered copy of this brain for offspring."""
        def mutate_matrix(mat):
            return [[w + random.uniform(-rate, rate) for w in row] for row in mat]
        def mutate_vector(vec):
            return [v + random.uniform(-rate, rate) for v in vec]

        new_W1 = mutate_matrix(self.W1)
        new_b1 = mutate_vector(self.b1)
        new_W2 = mutate_matrix(self.W2)
        new_b2 = mutate_vector(self.b2)
        
        return NeuralNet(self.input_size, self.hidden_size, self.output_size, (new_W1, new_b1, new_W2, new_b2))

class NeuralBug(BaseBug):
    """
    The NeuralBug is allowed to have an actual network of neurons, not just
    some hardcoded heuristics.
    """
    def __init__(self, vision_cone, brain=None):
        super().__init__(vision_cone)
        
        # The Directions array MUST stay in this exact order so the brain 
        # understands which input/output index matches which direction.
        self.directions = [
            "forward", "forward_left", "forward_right", 
            "left", "right", 
            "back_left", "back_right", "back"
        ]
        
        # 16 Inputs (8 food, 8 walls), 12 Hidden Neurons, 8 Outputs (Movement choices)
        if brain is None:
            self.brain = NeuralNet(input_size=16, hidden_size=12, output_size=8)
        else:
            self.brain = brain

    def request_action(self, perception):
        inputs = []
        
        # --- BUILD THE INPUT VECTOR (16 numbers) ---
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
                elif tile == WALL_CHAR and distance == 1:
                    wall_score = 1.0 # 1.0 means a wall is touching us
                    break
                    
            food_inputs.append(food_score)
            wall_inputs.append(wall_score)
            
        # Combine them into a single list of 16 numbers
        inputs = food_inputs + wall_inputs

        # --- THINK ---
        # Pass the 16 inputs into the neural network to get 8 output scores
        output_scores = self.brain.forward(inputs)

        # --- ACT ---
        # Find the index of the highest score, and return that direction
        best_index = output_scores.index(max(output_scores))

        return self.directions[best_index]

    def mutate(self, mutation_rate=0.05):
        # Mutate the neural network and pass it into a new child bug
        mutated_brain = self.brain.mutate(rate=mutation_rate)

        return mutated_brain


class Simulation:
    def __init__(self, world, bug, max_iterations=MAX_ITERATIONS, life_force=LIFE_FORCE):
        self.world = world
        self.bug = bug
        self.max_iterations = max_iterations
        self.life_force = life_force
        
    def run(self):
        """Runs the bug through the world until it starves or hits the iteration limit."""
        until_we_die = self.life_force
        turns_survived = 0
        
        for turn in range(self.max_iterations):
            until_we_die -= 1
            turns_survived += 1
            
            # 1. Look
            perception = self.world.get_perception(**self.bug.vision_cone)
            
            # 2. Think
            next_action = self.bug.request_action(perception=perception)
            
            # 3. Act
            move_result = self.world.move_relative(next_action)
            
            # 4. React
            if move_result == "food":
                until_we_die = self.life_force
        
            # 5. Check Survival
            if until_we_die <= 0:
                break 
                
        # Return standard metrics so the trainer knows what happened
        food_collected = getattr(self.world, 'food_collected', 0)
        
        return {
            "turns_survived": turns_survived,
            "food_collected": food_collected,
            "starved": until_we_die <= 0
        }

#
# Fitness Functions
#
# What does it mean to "win"?
#

def fitness_gluttony(world, turns_survived):
    """
    The 'Yo, food is good' mindset.
    Ignores how many turns it took, solely rewards the amount of food eaten.
    """
    return float(getattr(world, 'food_collected', 0))

def fitness_longevity(world, turns_survived):
    """
    The survivalist mindset.
    Rewards staying alive as long as possible. Food is only a means to an end.
    """
    return float(turns_survived)

def fitness_efficiency(world, turns_survived):
    """
    A hybrid mindset.
    Rewards eating food, but PENALIZES taking too long to do it. 
    """
    food = getattr(world, 'food_collected', 0)
    # 50 points per food, minus 1 point for every turn wasted
    return (food * 50.0) - turns_survived


def basic_bugs():
    # Build a list of tuples: (BugClass, ConeName, ConeDictionary)
    # This lets you pit different classes AND different eyes against each other
    competitors = [
        (RandomBug, "Balanced", VISION_CONES["Balanced"]),
        (RandomBug, "Tunnel", VISION_CONES["Tunnel"]),
        (RandomBug, "Prey", VISION_CONES["Prey"]),
        (RandomBug, "Radar", VISION_CONES["Radar"]),
        (ForwardBug, "Balanced", VISION_CONES["Balanced"]),
        (ForwardBug, "Tunnel", VISION_CONES["Tunnel"]),
        (ForwardBug, "Prey", VISION_CONES["Prey"]),
        (ForwardBug, "Radar", VISION_CONES["Radar"])
    ] 
    
    # This dictionary will store all the final stats
    tournament_results = {}
    
    print(f"STARTING TOURNAMENT: {NUM_EPOCHS} Epochs per Bug\n")
    starting_food = generate_initial_food()
    print(f"Starting with food {starting_food}")
    
    for BugClass, cone_name, cone_dict in competitors:
        # Create a unique name for the scoreboard (e.g., "ForwardBug (Tunnel)")
        bug_name = f"{BugClass.__name__} ({cone_name})"
        print(f"Testing {bug_name}...")
        
        # Trackers for this specific Bug type
        total_turns = 0
        total_food = 0
        max_food_in_one_run = 0
        starvations = 0

        # Run the epochs for this bug
        for _ in range(NUM_EPOCHS):
            world = World(initial_food=starting_food)
            
            # Pass the vision_cone dictionary directly into the bug on spawn
            bug = BugClass(vision_cone=cone_dict) 
            
            until_we_die = LIFE_FORCE
            
            for turn in range(MAX_ITERATIONS):
                until_we_die -= 1
                
                # The bug uses its own vision cone property
                perception = world.get_perception(**bug.vision_cone)
                next_action = bug.request_action(perception=perception)
                move_result = world.move_relative(next_action)
                
                if move_result == "food":
                    until_we_die = LIFE_FORCE
            
                if until_we_die <= 0:
                    starvations += 1
                    break 
                    
            # Tally stats at the end of the epoch
            food_eaten = getattr(world, 'food_collected', 0)
            
            total_turns += turn
            total_food += food_eaten
            if food_eaten > max_food_in_one_run:
                max_food_in_one_run = food_eaten

        # Calculate Averages for this Bug
        tournament_results[bug_name] = {
            "avg_turns": total_turns / NUM_EPOCHS,
            "avg_food": total_food / NUM_EPOCHS,
            "max_food": max_food_in_one_run,
            "survival_rate": ((NUM_EPOCHS - starvations) / NUM_EPOCHS) * 100
        }

    # --- Print the Side-by-Side Comparison ---
    print("\n=====================================================================")
    print(f" {'BUG TYPE & EYES':<25} | {'AVG TURNS':<10} | {'AVG FOOD':<10} | {'MAX FOOD':<8}")
    print("=====================================================================")
    
    for bug_name, stats in tournament_results.items():
        print(f" {bug_name:<25} | {stats['avg_turns']:<10.1f} | {stats['avg_food']:<10.2f} | {stats['max_food']:<8}")
        
    print("=====================================================================\n")


def train_genetic_algorithm(generations=GENERATIONS, population_size=POP_SIZE, mutation_rate=MUTATION_RATE):
    print(f"--- STARTING EVOLUTION ---")
    print(f"Generations: {generations} | Population: {population_size} | Mutation Rate: {mutation_rate}\n")
    
    # We will use the 'Radar' vision cone for this test so they have 360 awareness
    training_cone = VISION_CONES["Radar"]
    
    # 1. Initialize the first generation with completely random brains
    population = [BrainBug(vision_cone=training_cone) for _ in range(population_size)]
    
    for gen in range(generations):
        # Generate a new random food layout for this generation so they don't just memorize one map
        starting_food = generate_initial_food()
        
        # 2. Evaluate Fitness (Run the tournament for every bug)
        for bug in population:
            world = World(initial_food=starting_food)
            until_we_die = LIFE_FORCE
            
            for turn in range(MAX_ITERATIONS):
                until_we_die -= 1
                
                perception = world.get_perception(**bug.vision_cone)
                next_action = bug.request_action(perception=perception)
                move_result = world.move_relative(next_action)
                
                if move_result == "food":
                    until_we_die = LIFE_FORCE
            
                if until_we_die <= 0:
                    break 
                    
            # Save the score directly onto the bug object
            bug.fitness = getattr(world, 'food_collected', 0)

        # 3. Sort the population by fitness (highest food eaten at the top)
        population.sort(key=lambda b: getattr(b, 'fitness', 0), reverse=True)
        
        # Calculate some stats for the console
        top_score = population[0].fitness
        avg_score = sum(b.fitness for b in population) / population_size
        
        print(f"Gen {gen + 1:03d} | Top Food: {top_score:<4} | Avg Food: {avg_score:.2f} | Best Genes: "
              f"Food({population[0].genes['food_weight']:.2f}) "
              f"Wall({population[0].genes['wall_weight']:.2f}) "
              f"Empty({population[0].genes['empty_weight']:.2f})")

        # 4. Selection and Mutation (Breed the next generation)
        # Keep the top 10% of bugs as the "parents"
        num_parents = max(2, population_size // 10)
        parents = population[:num_parents]
        
        next_generation = []
        
        # Elitism: Keep the absolute best bugs exactly as they are so we never lose our progress
        next_generation.extend(parents)
        
        # Fill the rest of the population with mutated children
        while len(next_generation) < population_size:
            # Pick a random successful parent
            parent = random.choice(parents)
            
            # Get mutated genes
            child_genes = parent.mutate(mutation_rate=mutation_rate)
            
            # Create the child and add it to the new population
            child_bug = BrainBug(vision_cone=training_cone, genes=child_genes)
            next_generation.append(child_bug)
            
        # Replace the old population with the new one
        population = next_generation

    print("\n--- EVOLUTION COMPLETE ---")
    best_bug = population[0]
    print(f"Genes:")
    print(f"Food  Weight: {best_bug.genes['food_weight']:.4f}")
    print(f"Wall  Weight: {best_bug.genes['wall_weight']:.4f}")
    print(f"Empty Weight: {best_bug.genes['empty_weight']:.4f}")
    
    return best_bug

def train_neural_algorithm(generations=GENERATIONS, population_size=POP_SIZE, mutation_rate=MUTATION_RATE):
    Log.info("--- STARTING NEURAL EVOLUTION ---")
    Log.info(f"Generations: {generations} | Population: {population_size} | Mutation Rate: {mutation_rate}\n")
    
    training_cone = VISION_CONES["Radar"]
    
    # Initialize with NeuralBug
    population = [NeuralBug(vision_cone=training_cone) for _ in range(population_size)]
    
    # Track the best score across ALL generations to log breakthroughs
    overall_best_score = -1
    
    for gen in range(generations):
        starting_food = generate_initial_food()
        
        for bug in population:
            world = World(initial_food=starting_food)
            until_we_die = LIFE_FORCE
            
            for turn in range(MAX_ITERATIONS):
                until_we_die -= 1
                
                perception = world.get_perception(**bug.vision_cone)
                next_action = bug.request_action(perception=perception)
                move_result = world.move_relative(next_action)
                
                if move_result == "food":
                    until_we_die = LIFE_FORCE
            
                if until_we_die <= 0:
                    break 
                    
            bug.fitness = getattr(world, 'food_collected', 0)

        population.sort(key=lambda b: getattr(b, 'fitness', 0), reverse=True)
        
        top_score = population[0].fitness
        avg_score = sum(b.fitness for b in population) / population_size
        
        # Standard generation heartbeat log
        print(f"Gen {gen + 1:03d} | Top Food: {top_score:<4} | Avg Food: {avg_score:.2f}")

        # Announce when the AI breaks a new ceiling
        if top_score > overall_best_score:
            overall_best_score = top_score
            Log.debug(f"New Apex Bug evolved in Gen %s with a score of %s!", gen + 1, top_score)

        # 4. Selection and Mutation
        num_parents = max(2, population_size // 10)
        parents = population[:num_parents]
        
        next_generation = []
        next_generation.extend(parents)
        
        while len(next_generation) < population_size:
            parent = random.choice(parents)
            
            child_brain = parent.mutate(mutation_rate=mutation_rate)
            child_bug = NeuralBug(vision_cone=training_cone, brain=child_brain)
            next_generation.append(child_bug)
            
        population = next_generation

    Log.info("\n--- NEURAL EVOLUTION COMPLETE ---")
    best_bug = population[0]
    Log.info(f"Final Apex Bug reached a top score of {best_bug.fitness}")
    
    return best_bug


def train_neural_algorithm_with_trials(
        generations=GENERATIONS, 
        population_size=POP_SIZE, 
        mutation_rate=MUTATION_RATE, 
        trials=3
        ):
    Log.info("--- STARTING MULTI-TRIAL NEURAL EVOLUTION ---")
    Log.info(f"Generations: {generations} | Population: {population_size} | Trials per Bug: {trials} | Mutation Rate: {mutation_rate}\n")
    
    # We will use the 'Radar' vision cone for this test so they have 360 awareness
    training_cone = VISION_CONES["Radar"]
    
    # 1. Initialize the first generation with completely random brains
    population = [NeuralBug(vision_cone=training_cone) for _ in range(population_size)]
    
    # Track the best score across ALL generations to log breakthroughs
    overall_best_score = -1
    
    for gen in range(generations):
        
        # 2. Evaluate Fitness (Run the multi-trial tournament for every bug)
        for bug in population:
            total_score_across_trials = 0
            
            for _ in range(trials):
                # Setup the board and the rules
                world = World(initial_food=generate_initial_food())
                sim = Simulation(world, bug)
                
                # Run it
                results = sim.run()
                
                # Evaluate it
                trial_score = fitness_gluttony(world, results["food_collected"])
                total_fitness_across_trials += trial_score
                
            # Fitness is the exact AVERAGE performance across all map layouts
            bug.fitness = total_score_across_trials / trials

        # 3. Sort the population by true average fitness
        population.sort(key=lambda b: getattr(b, 'fitness', 0), reverse=True)
        
        # Calculate stats for the console
        top_score = population[0].fitness
        avg_score = sum(b.fitness for b in population) / population_size
        
        # Standard generation heartbeat log
        print(f"Gen {gen + 1:03d} | Top Avg Food: {top_score:<5.1f} | Pop Avg Food: {avg_score:.2f}")

        # Announce when the AI breaks a new ceiling
        if top_score > overall_best_score:
            overall_best_score = top_score
            Log.info(f"New Top Bug evolved in Gen {gen + 1} with a true average score of {top_score:.1f}!")

        # 4. Selection and Mutation
        num_parents = max(2, population_size // 10)
        parents = population[:num_parents]
        
        next_generation = []
        
        # Elitism: Keep the absolute best bugs exactly as they are
        next_generation.extend(parents)
        
        # Fill the rest of the population with mutated children
        while len(next_generation) < population_size:
            parent = random.choice(parents)
            
            child_brain = parent.mutate(mutation_rate=mutation_rate)
            child_bug = NeuralBug(vision_cone=training_cone, brain=child_brain)
            next_generation.append(child_bug)
            
        population = next_generation

    Log.info("\n--- NEURAL EVOLUTION COMPLETE ---")
    best_bug = population[0]
    Log.info(f"Final Bug reached a top average score of {best_bug.fitness:.1f}")
    
    return best_bug

def train_neural_algorithm_with_trials_and_fitness(
    generations=GENERATIONS, 
    population_size=POP_SIZE, 
    mutation_rate=MUTATION_RATE, 
    trials=3,
    fitness_fn=fitness_gluttony
):
    # Dynamically log which fitness function is driving this evolution
    fitness_name = fitness_fn.__name__.upper()
    Log.info(f"--- STARTING NEURAL EVOLUTION ({fitness_name}) ---")
    Log.info(f"Generations: {generations} | Population: {population_size} | Trials: {trials}\n")
    
    training_cone = VISION_CONES["Radar"]
    population = [NeuralBug(vision_cone=training_cone) for _ in range(population_size)]
    overall_best_score = -9999999 
    
    for gen in range(generations):
        
        for bug in population:
            total_fitness_across_trials = 0
            
            for _ in range(trials):
                # Setup the board and the rules
                world = World(initial_food=generate_initial_food())
                sim = Simulation(world, bug)
                
                # Run it
                results = sim.run()
                
                # Evaluate it
                trial_score = fitness_fn(world, results["turns_survived"])
                total_fitness_across_trials += trial_score
                
            bug.fitness = total_fitness_across_trials / trials

        population.sort(key=lambda b: getattr(b, 'fitness', -99999), reverse=True)
        
        top_score = population[0].fitness
        avg_score = sum(b.fitness for b in population) / population_size
        
        print(f"Gen {gen + 1:03d} | Top Score: {top_score:<7.1f} | Avg Score: {avg_score:.2f}")

        if top_score > overall_best_score:
            overall_best_score = top_score
            Log.info(f"New Best Bug! ({fitness_name} Score: {top_score:.1f})")

        # 4. Selection and Mutation
        num_parents = max(2, population_size // 10)
        parents = population[:num_parents]
        
        next_generation = []
        
        # Elitism: Keep the absolute best bugs exactly as they are
        next_generation.extend(parents)
        
        # Fill the rest of the population with mutated children
        while len(next_generation) < population_size:
            parent = random.choice(parents)
            
            child_brain = parent.mutate(mutation_rate=mutation_rate)
            child_bug = NeuralBug(vision_cone=training_cone, brain=child_brain)
            next_generation.append(child_bug)
            
        population = next_generation


    Log.info("\n--- NEURAL EVOLUTION COMPLETE ---")
    best_bug = population[0]
    Log.info(f"Final Bug reached a top average score of {best_bug.fitness:.1f}")
    
    return best_bug

if __name__ == "__main__":
    #
    # Basic Bugs tests the Rules based bugs
    #
    basic_bugs()

    #
    # Genetic Algorithm is simple weights and outputs.
    # Think of them like a cell: it gets input and can
    # decide to move somewhere but that's it.
    #
    # train_genetic_algorithm()

    #
    # Neural Networks
    # Now we are getting into the fun stuff. This gets to
    # have a real "brain", with hidden "layers", activation,
    # all those magic words. 
    #
    # train_neural_algorithm()
    # train_neural_algorithm_with_trials()

    #
    # Train with Trials and Fitness
    #
    # allows you to give it a custom fitness function and runs the
    # Bug through generations of testing, driving by args or CONSTS above
    #
    train_neural_algorithm_with_trials_and_fitness(fitness_fn=fitness_gluttony)
    train_neural_algorithm_with_trials_and_fitness(fitness_fn=fitness_longevity)
    train_neural_algorithm_with_trials_and_fitness(fitness_fn=fitness_efficiency)
