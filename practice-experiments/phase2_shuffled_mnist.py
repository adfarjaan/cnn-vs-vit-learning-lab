"""
Phase 2: Shuffled MNIST Test
Tests how CNN vs ViT handle shuffled spatial structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.cnn import SimpleCNN
from models.vit import SimpleViT
from utils.datasets import ShuffledMNIST
from utils.train import train_model

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*50)
print("PHASE 2: Shuffled MNIST Test")
print("="*50)

# Create datasets
print("\nCreating MNIST datasets...")

# Normal MNIST (no shuffle)
normal_train = ShuffledMNIST(train=True, shuffle=False)
normal_test = ShuffledMNIST(train=False, shuffle=False)

# Shuffled MNIST (quadrants shuffled)
shuffled_train = ShuffledMNIST(train=True, shuffle=True, shuffle_prob=1.0)
shuffled_test = ShuffledMNIST(train=False, shuffle=True, shuffle_prob=1.0)

# Create dataloaders
normal_train_loader = DataLoader(normal_train, batch_size=128, shuffle=True)
normal_test_loader = DataLoader(normal_test, batch_size=128, shuffle=False)
shuffled_train_loader = DataLoader(shuffled_train, batch_size=128, shuffle=True)
shuffled_test_loader = DataLoader(shuffled_test, batch_size=128, shuffle=False)

# Visualize samples
print("\nVisualizing normal vs shuffled MNIST...")
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i in range(4):
    # Normal
    img, label = normal_train[i]
    axes[0, i].imshow(img.squeeze(), cmap='gray')
    axes[0, i].set_title(f"Normal: {label}")
    axes[0, i].axis('off')
    
    # Shuffled
    img, label = shuffled_train[i]
    axes[1, i].imshow(img.squeeze(), cmap='gray')
    axes[1, i].set_title(f"Shuffled: {label}")
    axes[1, i].axis('off')
plt.suptitle("Phase 2: Normal vs Shuffled MNIST", fontsize=14)
plt.tight_layout()
plt.savefig('results/phase2_samples.png', dpi=150)
plt.show()

# ============================================================
# EXPERIMENT 1: CNN on Normal MNIST
# ============================================================
print("\n" + "-"*30)
print("Experiment 1: CNN on Normal MNIST")
print("-"*30)
cnn_normal = SimpleCNN(input_channels=1, num_classes=10)
cnn_normal_loss, cnn_normal_train, cnn_normal_test = train_model(
    cnn_normal, normal_train_loader, normal_test_loader, epochs=5, lr=0.001, device=device
)
cnn_normal_final = cnn_normal_test[-1]

# ============================================================
# EXPERIMENT 2: CNN on Shuffled MNIST
# ============================================================
print("\n" + "-"*30)
print("Experiment 2: CNN on Shuffled MNIST")
print("-"*30)
cnn_shuffled = SimpleCNN(input_channels=1, num_classes=10)
cnn_shuffled_loss, cnn_shuffled_train, cnn_shuffled_test = train_model(
    cnn_shuffled, shuffled_train_loader, shuffled_test_loader, epochs=5, lr=0.001, device=device
)
cnn_shuffled_final = cnn_shuffled_test[-1]

# ============================================================
# EXPERIMENT 3: ViT on Normal MNIST
# ============================================================
print("\n" + "-"*30)
print("Experiment 3: ViT on Normal MNIST")
print("-"*30)

class AdaptedViT_MNIST(SimpleViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_embed.proj = torch.nn.Conv2d(1, 64, kernel_size=4, stride=4)

vit_normal = AdaptedViT_MNIST(img_size=28, patch_size=4, num_classes=10)
vit_normal_loss, vit_normal_train, vit_normal_test = train_model(
    vit_normal, normal_train_loader, normal_test_loader, epochs=5, lr=0.001, device=device
)
vit_normal_final = vit_normal_test[-1]

# ============================================================
# EXPERIMENT 4: ViT on Shuffled MNIST
# ============================================================
print("\n" + "-"*30)
print("Experiment 4: ViT on Shuffled MNIST")
print("-"*30)
vit_shuffled = AdaptedViT_MNIST(img_size=28, patch_size=4, num_classes=10)
vit_shuffled_loss, vit_shuffled_train, vit_shuffled_test = train_model(
    vit_shuffled, shuffled_train_loader, shuffled_test_loader, epochs=5, lr=0.001, device=device
)
vit_shuffled_final = vit_shuffled_test[-1]

# ============================================================
# PLOT RESULTS
# ============================================================
print("\nPlotting comparison results...")

fig, ax = plt.subplots(figsize=(10, 6))
models = ['CNN\nNormal', 'CNN\nShuffled', 'ViT\nNormal', 'ViT\nShuffled']
accuracies = [cnn_normal_final, cnn_shuffled_final, vit_normal_final, vit_shuffled_final]
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Phase 2: CNN vs ViT on Normal vs Shuffled MNIST', fontsize=14)
ax.set_ylim(0, 105)

# Add values on top of bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.grid(axis='y', alpha=0.3)

# Add conclusion
drop_cnn = cnn_normal_final - cnn_shuffled_final
drop_vit = vit_normal_final - vit_shuffled_final

conclusion = f" CNN drop: {drop_cnn:.1f}%\n ViT drop: {drop_vit:.1f}%"
ax.text(0.5, -0.2, conclusion,
        ha='center', va='center', transform=ax.transAxes, 
        fontsize=11, style='italic', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray'))

plt.tight_layout()
plt.savefig('results/phase2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*50)
print("PHASE 2 COMPLETE!")
print("="*50)
print(f"\n Results Summary:")
print(f"   CNN on Normal MNIST:     {cnn_normal_final:.2f}%")
print(f"   CNN on Shuffled MNIST:   {cnn_shuffled_final:.2f}%")
print(f"   ViT on Normal MNIST:     {vit_normal_final:.2f}%")
print(f"   ViT on Shuffled MNIST:   {vit_shuffled_final:.2f}%")

print(f"\nAccuracy Drop:")
print(f"   CNN drop: {drop_cnn:.2f}%")
print(f"   ViT drop: {drop_vit:.2f}%")

if drop_cnn > drop_vit:
    print("\n CONCLUSION: CNN is MORE affected by shuffling.")
    print("   This proves CNNs rely on LOCAL spatial structure.")
    print("   ViTs use GLOBAL attention and are more robust.")
elif drop_vit > drop_cnn:
    print("\n CONCLUSION: ViT is MORE affected by shuffling.")
else:
    print("\n Both models were similarly affected.")

print("\n Results saved in 'results/phase2_comparison.png'")
print("="*50)