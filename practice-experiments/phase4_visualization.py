"""
Phase 4: Model Visualization
Visualize CNN feature maps and ViT attention maps side-by-side
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from models.cnn import SimpleCNN
from models.vit import SimpleViT

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*50)
print("PHASE 4: Model Visualization")
print("="*50)

# Load a sample image from CIFAR-10
print("\nLoading sample image...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Get a test image
image, label = testset[0]
image = image.unsqueeze(0).to(device)  # Add batch dimension

# Class names for CIFAR-10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Sample image: {classes[label]}")

# ==========================================
# CNN Visualization: Extract Feature Maps
# ==================================
print("\n" + "-"*30)
print("Visualizing CNN Feature Maps...")
print("-"*30)

# Create CNN model
cnn = SimpleCNN(input_channels=3, num_classes=10).to(device)

# Hook to extract feature maps
cnn_features = []
def hook_fn(module, input, output):
    cnn_features.append(output.detach().cpu())

# Register hooks on conv layers
hooks = []
for name, layer in cnn.named_children():
    if isinstance(layer, torch.nn.Conv2d):
        hooks.append(layer.register_forward_hook(hook_fn))

# Forward pass
with torch.no_grad():
    cnn(image)

# Remove hooks
for hook in hooks:
    hook.remove()

# Visualize feature maps
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i in range(min(8, len(cnn_features))):
    row, col = i // 4, i % 4
    if i < len(cnn_features):
        # Take first 4 channels of each layer
        fmaps = cnn_features[i][0, :4, :, :].numpy()
        # Normalize for visualization
        fmaps = (fmaps - fmaps.min()) / (fmaps.max() - fmaps.min() + 1e-8)
        # Show as grid
        combined = np.concatenate([fmaps[j] for j in range(4)], axis=1)
        axes[row, col].imshow(combined, cmap='viridis')
        axes[row, col].set_title(f'Conv Layer {i+1}')
        axes[row, col].axis('off')

# Hide unused subplots
for i in range(8, 12):
    row, col = i // 4, i % 4
    axes[row, col].axis('off')

plt.suptitle("CNN Feature Maps (First 4 channels per layer)", fontsize=14)
plt.tight_layout()
plt.savefig('results/phase4_cnn_features.png', dpi=150)
plt.show()

# =======================
# ViT Visualization: Extract Attention Maps
# ==================================
print("\n" + "-"*30)
print("Visualizing ViT Attention Maps...")
print("-"*30)

# Create ViT model
vit = SimpleViT(img_size=32, patch_size=4, num_classes=10).to(device)

# Forward pass with attention
with torch.no_grad():
    _, attn_maps = vit(image, return_attention=True)

# Get attention from last layer
last_attn = attn_maps[-1].cpu()  # (B, heads, N, N)
cls_attn = last_attn[0, :, 0, 1:].mean(0)  # Average over heads
num_patches = 8  # 32/4 = 8
attn_grid = cls_attn.reshape(num_patches, num_patches).numpy()

# Visualize attention grid
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(attn_grid, cmap='hot', interpolation='nearest')
ax.set_title("ViT Attention Map (CLS token to patches)")
ax.set_xlabel("Patch Column")
ax.set_ylabel("Patch Row")
plt.colorbar(im)
plt.tight_layout()
plt.savefig('results/phase4_vit_attention.png', dpi=150)
plt.show()

# ============================================================
# Side-by-Side Comparison
# ==========================
print("\n" + "-"*30)
print("Creating Side-by-Side Comparison...")
print("-"*30)

# Original image
img_np = image[0].cpu().numpy().transpose(1, 2, 0)
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

# CNN feature map (last layer)
last_cnn = cnn_features[-1][0]  # (64, 4, 4)
last_cnn_avg = last_cnn.mean(dim=0).numpy()
# Upscale to 32x32
cnn_heatmap = zoom(last_cnn_avg, (32/4, 32/4))

# ViT attention map
vit_heatmap = zoom(attn_grid, (32/8, 32/8))

# Plot side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].imshow(img_np)
axes[0].set_title(f"Original Image\nClass: {classes[label]}")
axes[0].axis('off')

# CNN
axes[1].imshow(img_np, alpha=0.6)
im1 = axes[1].imshow(cnn_heatmap, cmap='jet', alpha=0.4)
axes[1].set_title("CNN: Local Feature Focus\n(Sees edges & textures)")
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# ViT
axes[2].imshow(img_np, alpha=0.6)
im2 = axes[2].imshow(vit_heatmap, cmap='jet', alpha=0.4)
axes[2].set_title("ViT: Global Attention Focus\n(Sees relationships)")
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle("Phase 4: CNN Feature Maps vs ViT Attention Maps", fontsize=14)
plt.tight_layout()
plt.savefig('results/phase4_comparison.png', dpi=150)
plt.show()

# ======================
# Summary
# =====================
print("\n" + "="*50)
print("PHASE 4 COMPLETE!")
print("="*50)
print("\n What you just saw:")
print("   🔍 CNN: Focuses on local textures and edges")
print("   🌍 ViT: Looks at global relationships between patches")
print("\n Results saved:")
print("   - phase4_cnn_features.png")
print("   - phase4_vit_attention.png")
print("   - phase4_comparison.png")
print("="*50)