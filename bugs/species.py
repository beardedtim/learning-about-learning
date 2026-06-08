import torch
import torch.nn as nn

class GeneticColonySpecies(nn.Module):
    def __init__(self, num_sensors, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        input_dim = num_sensors + 1
        output_dim = 3 
        
        # 1. Vision Encoder
        self.vision_enc = nn.Linear(input_dim, hidden_dim)
        
        # 2. The Update Gate
        self.gate_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        # Inductive bias: Start by relying on vision (gate near 0.0)
        nn.init.constant_(self.gate_layer.bias, -1.0) 
        
        # 3. The Candidate Thought
        self.candidate_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 4. Action Decoder
        self.fc_action = nn.Linear(hidden_dim, output_dim)

    def forward(self, observation, memory):
        # 1. Process raw sight (SiLU preserves more genetic information than ReLU)
        vision_features = torch.nn.functional.silu(self.vision_enc(observation))
        
        combined_state = torch.cat([vision_features, memory], dim=-1)
        
        # 2. Compute the Gate 
        gate = torch.sigmoid(self.gate_layer(combined_state))
        
        # 3. Compute the new Candidate Thought (Tanh prevents exploding memory)
        candidate_memory = torch.tanh(self.candidate_layer(combined_state))
        
        # 4. Interpolate between past and present
        new_memory = (gate * memory) + ((1.0 - gate) * candidate_memory)
        
        # 5. Decide the action
        logits = self.fc_action(new_memory)
        action = torch.argmax(logits, dim=-1)
        
        return action, new_memory