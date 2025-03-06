import torch
import torch.nn as nn

class KnowledgeTransfer(nn.Module):
    def __init__(self, hidden_dim):
        super(KnowledgeTransfer, self).__init__()
        
        # Transfer layers
        self.short_to_medium = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.medium_to_long = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Attention for knowledge fusion
        self.fusion_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, short_mem, medium_mem, long_mem):
        # Knowledge transfer from short to medium memory
        s2m = self.short_to_medium(short_mem)
        medium_mem = medium_mem + s2m
        
        # Knowledge transfer from medium to long memory
        m2l = self.medium_to_long(medium_mem)
        long_mem = long_mem + m2l
        
        # Fusion through attention
        query = long_mem
        key = torch.cat([short_mem, medium_mem, long_mem], dim=1)
        value = key
        
        fused_mem, _ = self.fusion_attention(query, key, value)
        fused_mem = self.norm(fused_mem + query)
        
        return fused_mem 