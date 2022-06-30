import torch
from torch import nn
from torch.nn import LayerNorm
from att import Attention
from ff import FeedForward
from einops import rearrange

class ViT(nn.Module):
    def __init__(self, depth, num_patches, patch_size, patch_dim, latent_dim, mlp_dim, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.patch_to_embed = nn.Linear(patch_dim, latent_dim)
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + 1, latent_dim))
        self.cls = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.layers = nn.ModuleList([])
        
        self.classification_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

        for _ in range(depth):
            norm = LayerNorm(latent_dim)
            att = Attention(latent_dim)
            ff = FeedForward(latent_dim, mlp_dim)
            self.layers.append(nn.ModuleList([norm, att, ff]))

    def forward(self, x):
        batch_size = x.shape[0]
        patch = rearrange(x, "b c (ph h) (pw w) -> b (h w) (ph pw c)", ph=self.patch_size, pw=self.patch_size)
        embed = self.patch_to_embed(patch)
        cls_tokens = self.cls.repeat(batch_size, 1, 1)
        x = torch.cat((cls_tokens, embed), dim=1)

        for norm, att, ff in self.layers:
            x = att(norm(x)) + x
            x = ff(x) + x
        
        cls_tokens = x[:, 0]
        return self.classification_head(cls_tokens)

img_size = 32
patch_size = 4
num_patches = (img_size // patch_size) ** 2
patch_dim = 3 * patch_size ** 2
depth = 12

x = torch.randn(256, 3, 32, 32)
vit = ViT(depth, num_patches, patch_size, patch_dim, latent_dim=64, mlp_dim=512, num_classes=10)