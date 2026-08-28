"""
Phase 1: Synthetic Lines Test
Compares CNN vs ViT on simple vertical/horizontal line classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.cnn import SimpleCNN
from models.vit import SimpleViT
from utils.datasets import SyntheticLinesDataset
from utils.train import train_model

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*50)
print("PHASE 1: Synthetic Lines Test")
print("="*50)

# Create dataset
print("\nCreating synthetic dataset...")
dataset = SyntheticLinesDataset(num_samples=2000)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Visualize samples
print("\nVisualizing sample images...")
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    img, label = dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"{'Vertical' if label==0 else 'Horizontal'}")
    ax.axis('off')
plt.suptitle("Phase 1: Synthetic Lines Dataset")
plt.tight_layout()
plt.savefig('results/phase1_samples.png')
plt.show()

# Train CNN
print("\n" + "-"*30)
print("Training CNN on synthetic lines...")
print("-"*30)
cnn = SimpleCNN(input_channels=1, num_classes=2)
cnn_losses, cnn_train_accs, cnn_test_accs = train_model(
    cnn, train_loader, test_loader, epochs=5, lr=0.01, device=device
)

# Train ViT (adapted for 1-channel input)
print("\n" + "-"*30)
print("Training ViT on synthetic lines...")
print("-"*30)

class AdaptedViT(SimpleViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override patch embedding for 1 channel
        self.patch_embed.proj = torch.nn.Conv2d(1, 64, kernel_size=4, stride=4)

vit = AdaptedViT(img_size=16, patch_size=4, num_classes=2)
vit_losses, vit_train_accs, vit_test_accs = train_model(
    vit, train_loader, test_loader, epochs=5, lr=0.01, device=device
)

# Plot comparison
print("\nPlotting results...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training accuracy
axes[0].plot(range(1, 6), cnn_train_accs, 'o-', label='CNN', linewidth=2, markersize=8)
axes[0].plot(range(1, 6), vit_train_accs, 's-', label='ViT', linewidth=2, markersize=8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Accuracy (%)')
axes[0].set_title('Training Speed Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test accuracy
axes[1].plot(range(1, 6), cnn_test_accs, 'o-', label='CNN', linewidth=2, markersize=8)
axes[1].plot(range(1, 6), vit_test_accs, 's-', label='ViT', linewidth=2, markersize=8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Accuracy (%)')
axes[1].set_title('Test Accuracy Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Phase 1: CNN vs ViT on Synthetic Lines", fontsize=14)
plt.tight_layout()
plt.savefig('results/phase1_comparison.png', dpi=150)
plt.show()

print("\n" + "="*50)
print("Phase 1 Complete!")
print(f"CNN Final Test Accuracy: {cnn_test_accs[-1]:.2f}%")
print(f"ViT Final Test Accuracy: {vit_test_accs[-1]:.2f}%")
print("="*50)