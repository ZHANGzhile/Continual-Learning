import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-Head Attention
        x_norm = self.norm1(x)
        attended, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attended
        
        # Feed Forward
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + ff_out
        
        return x

class ShortMemoryBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers=2, dropout=0.1):
        super(ShortMemoryBlock, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # LSTM层用于捕获短期依赖
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Transformer处理
        for layer in self.layers:
            x = layer(x)
            
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        x = x + self.norm(lstm_out)
        
        return x

class MediumMemoryBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers=4, dropout=0.1):
        super(MediumMemoryBlock, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 额外的注意力机制用于捕获中期依赖
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # 保存输入用于cross attention
        identity = x
        
        # Transformer处理
        for layer in self.layers:
            x = layer(x)
            
        # Cross Attention处理
        cross_out, _ = self.cross_attention(x, identity, identity)
        x = x + self.norm(cross_out)
        
        return x

class LongTermMemoryBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers=6, dropout=0.1):
        super(LongTermMemoryBlock, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 记忆增强机制
        self.memory_size = 1024
        self.memory_dim = hidden_dim
        self.memory = nn.Parameter(torch.randn(self.memory_size, self.memory_dim))
        self.memory_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Transformer处理
        for layer in self.layers:
            x = layer(x)
            
        # 记忆增强处理
        batch_size = x.size(0)
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        mem_out, _ = self.memory_attention(x, memory, memory)
        x = x + self.memory_norm(mem_out)
        
        return x 