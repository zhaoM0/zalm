import torch 
import torch.nn as nn 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(2 / torch.pi) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
        
class FeedForward(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim),
            GELU(),
            nn.Linear(4 * in_dim, in_dim)
        )
    
    def forward(self, x):
        return self.layers(x)