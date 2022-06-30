import torch
from torch import nn
from torch.nn import LayerNorm
from att import Attention
from ff import FeedForward
from einops import rearrange

class ViT(nn.Module):
    def __init__(self, depth, num_patches, patch_size, patch_dim, embed_dim, mlp_dim, num_classes, dropout=0):
        super().__init__()
        self.patch_size = patch_size
        self.patch_to_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layers = nn.ModuleList([])
        
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        for _ in range(depth):
            norm = LayerNorm(embed_dim)
            att = Attention(embed_dim, num_heads=8, dim_head=64, dropout=dropout)
            ff = FeedForward(embed_dim, mlp_dim, dropout)
            self.layers.append(nn.ModuleList([norm, att, ff]))

    def forward(self, x):
        batch_size = x.shape[0]
        patch = rearrange(x, "b c (ph h) (pw w) -> b (h w) (ph pw c)", ph=self.patch_size, pw=self.patch_size)
        embed = self.patch_to_embed(patch)
        cls_tokens = self.cls.repeat(batch_size, 1, 1)
        x = torch.cat((cls_tokens, embed), dim=1)
        x += self.pos_embedding

        for norm, att, ff in self.layers:
            x = att(norm(x)) + x
            x = ff(x) + x
        
        cls_tokens = x[:, 0]
        return self.classification_head(cls_tokens)