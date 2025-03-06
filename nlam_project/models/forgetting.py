import torch
import torch.nn as nn

class ControlledForgetting(nn.Module):
    def __init__(self, memory_size, feature_dim):
        super(ControlledForgetting, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.forget_gate = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        forget_weights = self.forget_gate(x)
        return x * forget_weights 