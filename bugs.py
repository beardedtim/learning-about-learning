from log import Log
import random
import json
import numpy as np

import torch
import torch.nn as nn
import copy
from world import  FOOD_CHAR, RELATIVE_DIRECTIONS, WALL_CHAR


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
        sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in state['state_dict'].items()}
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
        # Use the last timestep's output explicitly to avoid accidental
        # removal of dimensions when hidden_size == 1 (out.squeeze() is unsafe).
        last = out[:, -1, :]                 # shape: (batch=1, hidden_size)
        logits = self.head(last).squeeze(0)  # shape: (output_size,)
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
        # We create a clone and apply in-place perturbations so the original
        # brain remains unchanged.
        child = self.clone()

        # Small epsilon to avoid zero-scale when a parameter tensor is constant
        eps = 1e-6

        with torch.no_grad():
            for param in child.parameters():
                # Compute a mutation std proportional to the parameter's own
                # observed standard deviation. This scales mutation magnitude
                # to each parameter's typical scale so tiny parameters don't
                # get swamped by large absolute noise and large params get
                # reasonably proportionate perturbations.
                try:
                    p_std = float(param.std().item())
                except Exception:
                    p_std = 0.0

                noise_std = rate * (p_std + eps)

                # Generate noise and add it in-place. Optionally, in future
                # we could only mutate a subset of elements (sparsity) to
                # create sparse, targeted mutations.
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)

        return child
 
    def crossover(self, other):
        """
        Returns a new TorchBrain by flipping a coin for each *parameter tensor*
        (not each weight individually, which keeps layer structure coherent).
        Then mutates the result slightly.
        """
        child = self.clone()
        # Element-wise crossover: for each parameter tensor we flip a coin per
        # element to decide whether the child's weight comes from `self` or
        # from `other`. This allows mixing within a weight matrix instead of
        # swapping whole tensors, which preserves useful substructures and
        # increases genetic diversity across offspring.
        with torch.no_grad():
            for p_child, p_other in zip(child.parameters(), other.parameters()):
                # Ensure shapes match (they should, but be defensive)
                if p_child.data.shape != p_other.data.shape:
                    # Fallback to coarse-grained copy if shapes differ
                    if random.random() > 0.5:
                        p_child.data.copy_(p_other.data)
                    continue

                # Create a random boolean mask of the same shape as the param
                # True -> take value from child (self), False -> take from other
                mask = torch.rand_like(p_child.data) > 0.5

                # Perform element-wise selection and write it back in-place
                mixed = torch.where(mask, p_child.data, p_other.data)
                p_child.data.copy_(mixed)

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
        state = {k: torch.tensor(v, dtype=torch.float32) for k, v in d["state_dict"].items()}
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
        # Place the input on the same device as the brain's parameters to avoid
        # device-mismatch errors if the model is moved to GPU.
        try:
            device = next(self.brain.parameters()).device
        except StopIteration:
            device = None
        return torch.tensor(raw, dtype=torch.float32, device=device)
 
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
 