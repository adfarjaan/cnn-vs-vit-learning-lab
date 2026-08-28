"""
Simple Vision Transformer for image classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv layer with stride = patch_size does the patch splitting
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)   # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return out, attn


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP"""
    
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=4):
        super().__init__()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Attention
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
    
    def forward(self, x):
        # Attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class SimpleViT(nn.Module):
    """Vision Transformer for image classification"""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64, 
                 num_heads=4, num_layers=4, num_classes=10):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x, return_attention=False):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pass through transformer blocks
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            if return_attention:
                attn_maps.append(attn)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification from CLS token
        out = self.head(x[:, 0])
        
        if return_attention:
            return out, attn_maps
        return out


# Quick test
if __name__ == "__main__":
    model = SimpleViT(img_size=32, patch_size=4)
    test_input = torch.randn(1, 3, 32, 32)
    output = model(test_input)
    print(f"Output shape: {output.shape}")