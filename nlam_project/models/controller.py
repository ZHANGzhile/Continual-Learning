import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveController(nn.Module):
    def __init__(self, hidden_dim):
        super(AdaptiveController, self).__init__()
        
        # Feature Extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Gating mechanism
        self.gate_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 for short/medium/long memory
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Calculate gating weights
        gates = F.softmax(self.gate_weights(features), dim=-1)
        
        # Split gates for different memory blocks
        short_gate, medium_gate, long_gate = gates.chunk(3, dim=-1)
        
        return {
            'features': features,
            'gates': {
                'short': short_gate,
                'medium': medium_gate,
                'long': long_gate
            }
        } 