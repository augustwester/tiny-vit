from torch import nn
from torch.nn.functional import softmax
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, num_heads, dim_head, dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.W = nn.Linear(dim, 3 * num_heads * dim_head, bias=False)
        self.scale = 1 / dim_head ** 0.5
        self.out = nn.Linear(dim_head * num_heads, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        qkv = self.W(x)
        q, k, v = rearrange(qkv, "b t (k h d) -> k b h t d", k=3, h=self.num_heads)
        qkT = q @ k.permute(0, 1, 3, 2)
        att = softmax(qkT / self.scale, dim=-1) @ v
        cat = rearrange(att, "b h t d -> b t (h d)")
        return self.dropout(self.out(cat))