import torch
import torch.nn as nn

class GeneticColonySpecies(nn.Module):
    def __init__(self, num_bugs, num_sensors, hidden_dim=16, device="cuda"):
        super().__init__()
        self.num_bugs = num_bugs
        self.device = device
        self.hidden_dim = hidden_dim # Store this for the memory reset
        
        input_dim = num_sensors + 1
        output_dim = 3 
        
        # --- THE GENES ---
        
        # Layer 1 (Input -> Hidden)
        self.W1 = nn.Parameter(torch.randn(num_bugs, input_dim, hidden_dim, device=device) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(num_bugs, 1, hidden_dim, device=device))
        
        # The Memory Matrix (Hidden -> Hidden)
        # This allows the bug to look at its own thoughts from the previous frame
        self.W_rec = nn.Parameter(torch.randn(num_bugs, hidden_dim, hidden_dim, device=device) * 0.1)
        
        # Layer 2 (Hidden -> Output)
        self.W2 = nn.Parameter(torch.randn(num_bugs, hidden_dim, output_dim, device=device) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(num_bugs, 1, output_dim, device=device))

        # The actual memory bank
        self.memory = None

    def reset_memory(self):
        """Must be called at the start of every new generation or life cycle!"""
        self.memory = torch.zeros(self.num_bugs, 1, self.hidden_dim, device=self.device)

    def forward(self, observations):
        x = observations.unsqueeze(1)
        
        # Initialize memory on the very first step
        if self.memory is None:
            self.reset_memory()
        
        # --- BATCHED RECURRENT NEURAL NETWORK MATH ---
        
        # 1. What does the bug see right now?
        current_vision = torch.bmm(x, self.W1)
        
        # 2. What was the bug thinking a millisecond ago?
        past_thoughts = torch.bmm(self.memory, self.W_rec)
        
        # 3. Combine them! (Input + Memory + Bias)
        hidden = torch.relu(current_vision + past_thoughts + self.b1)
        
        # 4. Save this new thought for the NEXT frame
        self.memory = hidden.detach() 
        
        # 5. Decide the action based on the combined thoughts
        logits = torch.bmm(hidden, self.W2) + self.b2
        logits = logits.squeeze(1)
        
        actions = torch.argmax(logits, dim=1)
        
        return actions