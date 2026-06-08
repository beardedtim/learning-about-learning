import torch
import torch.nn as nn

class GeneticColonySpecies(nn.Module):
    def __init__(self, num_sensors, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        input_dim = num_sensors + 1
        output_dim = 3 
        
        # 1. Vision Encoder
        # Instead of dumping raw sensor data into memory, we process it into concepts first.
        self.vision_enc = nn.Linear(input_dim, hidden_dim)
        
        # 2. The Update Gate
        # Takes what it sees + what it remembers, and outputs a decision (0.0 to 1.0)
        self.gate_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 3. The Candidate Thought
        # What the bug *would* think if it decided to completely overwrite its memory
        self.candidate_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 4. Action Decoder
        self.fc_action = nn.Linear(hidden_dim, output_dim)

    def forward(self, observation, memory):
        # 1. Process raw sight into abstract visual features
        vision_features = torch.relu(self.vision_enc(observation))
        
        # Concatenate vision and past memory to evaluate them together
        combined_state = torch.cat([vision_features, memory], dim=-1)
        
        # 2. Compute the Gate (Sigmoid squashes the output to exactly [0, 1])
        # A value near 1.0 means "keep old memory". A value near 0.0 means "write new memory".
        gate = torch.sigmoid(self.gate_layer(combined_state))
        
        # 3. Compute the new Candidate Thought
        candidate_memory = torch.relu(self.candidate_layer(combined_state))
        
        # 4. The Magic Math: Interpolate between the past and the present
        new_memory = (gate * memory) + ((1.0 - gate) * candidate_memory)
        
        # 5. Decide the action based on this highly refined memory state
        logits = self.fc_action(new_memory)
        action = torch.argmax(logits, dim=-1)
        
        return action, new_memory