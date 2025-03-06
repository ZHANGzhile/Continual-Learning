import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .controller import AdaptiveController
from .knowledge_transfer import KnowledgeTransfer
from .memory_block import ShortMemoryBlock, MediumMemoryBlock, LongTermMemoryBlock

class NLAM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):
        super(NLAM, self).__init__()
        
        # Input Embedding and Positional Encoding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Controller (Gating)
        self.controller = AdaptiveController(hidden_dim)
        
        # Memory Blocks with different depths
        self.short_memory = ShortMemoryBlock(hidden_dim, num_heads, num_layers=2, dropout=dropout)
        self.medium_memory = MediumMemoryBlock(hidden_dim, num_heads, num_layers=4, dropout=dropout)
        self.long_memory = LongTermMemoryBlock(hidden_dim, num_heads, num_layers=6, dropout=dropout)
        
        # Knowledge Transfer
        self.knowledge_transfer = KnowledgeTransfer(hidden_dim)
        
        # Output layers
        self.add_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x):
        # Input Embedding with positional encoding
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        # Controller Processing (Gating)
        control = self.controller(x)
        features, gates = control['features'], control['gates']
        
        # Parallel Memory Processing with gating
        short_mem = self.short_memory(features) * gates['short'].unsqueeze(-1)
        medium_mem = self.medium_memory(features) * gates['medium'].unsqueeze(-1)
        long_mem = self.long_memory(features) * gates['long'].unsqueeze(-1)
        
        # Knowledge Transfer between memory blocks
        memory_output = self.knowledge_transfer(
            short_mem=short_mem,
            medium_mem=medium_mem,
            long_mem=long_mem
        )
        
        # Final Processing
        out = self.add_norm(memory_output + x)
        out = self.mlp(out)
        
        return out 