
class Simulation:
    def __init__(self, world, bug, max_iterations, life_force):
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
        food_collected_this_run = 0 # Track locally to prevent World state leakage

        for turn in range(self.max_iterations):
            # 1. Look (Bug sees its true energy state)
            perception = self.world.get_perception(**self.bug.vision_cone)
            
            # 2. Think
            next_action = self.bug.request_action(perception=perception)
            
            # 3. Act
            move_result = self.world.move_relative(next_action)
            
            # 4. Pay the Energy Cost
            self.bug.life_force -= 1
            turns_survived += 1
            
            # 5. React (Restore the bug's life force if it eats)
            # Recommend importing FOOD_CHAR and using it here if move_result returns characters
            if move_result == "food": 
                self.bug.life_force = self.bug.max_life_force
                food_collected_this_run += 1
        
            # 6. Check Survival
            if self.bug.life_force <= 0:
                break
                
        return {
            "turns_survived": turns_survived,
            "food_collected": food_collected_this_run,
            "starved": self.bug.life_force <= 0
        }