import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_rank


def rank0_print(*args, **kwargs):
    try:
        if get_rank() == 0:
            print(*args, **kwargs)
    except Exception:
        print(*args, **kwargs)


class RouterV7(nn.Module):
    """
    基于multi-head attention构建
    """
    def __init__(self, config, expert_nums):
        super().__init__()
        self.config = config
        self.expert_nums = expert_nums
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size, 
            num_heads=8, 
            dropout=0.05, 
            kdim=self.config.hidden_size,
            vdim=self.config.hidden_size,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 2, self.expert_nums),
        )
    
    def forward(self, inputs_embeds, attn_mask):
        x, _ = self.multihead_attn(
            query=inputs_embeds,
            key=inputs_embeds,
            value=inputs_embeds,
            key_padding_mask=attn_mask,
        )
        x = self.mlp(x)
        
        # return torch.softmax(x, dim=-1)
        return x