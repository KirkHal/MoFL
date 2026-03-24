import torch
import torch.nn as nn
from torch.distributed import get_rank


def rank0_print(*args, **kwargs):
    try:
        if get_rank() == 0:
            print(*args, **kwargs)
    except Exception:
        print(*args, **kwargs)



class Router(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, out_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x.to(previous_dtype)



class Top2Router(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.dropout(x)
        probs = nn.functional.softmax(x, dim=-1)

        top2_values, top2_indices = torch.topk(probs, 2, dim=-1)
        normalized_top2 = torch.softmax(top2_values, dim=-1)
        gating_distribution = torch.zeros_like(probs, dtype=torch.float32)
        gating_distribution.scatter_(-1, top2_indices, normalized_top2)

        return gating_distribution.to(previous_dtype)



class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)

        return torch.round(x)


