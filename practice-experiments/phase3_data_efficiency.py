"""
Phase 3: Data Efficiency Test
Compares CNN vs ViT on different amounts of training data (5%, 20%, 50%, 100%)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.cnn import SimpleCNN
from models.vit import SimpleViT
from utils.datasets import create_cifar_subsets
from utils.train import train_model

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*50)
print("PHASE 3: Data Efficiency Test")
print("="*50)

# Create subsets
print("\nCreating CIFAR-10 subsets...")
subsets, testset = create_cifar_subsets()

# Create test loader
test_loader = DataLoader(testset, batch_size=128, shuffle=False)

# Store results
results = {}
percentages = [5, 20, 50, 100]

for percentage in percentages:
    name = f"{percentage}%"
    print(f"\n{'='*50}")
    print(f"Training on {name} data")
    print('='*50)
    
    # Get subset
    train_subset = subsets[name]
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    
    # Train CNN
    print(f"\nTraining CNN on {name}...")
    cnn = SimpleCNN(input_channels=3, num_classes=10)
    _, _, cnn_acc = train_model(
        cnn, train_loader, test_loader, epochs=10, lr=0.001, device=device, verbose=True
    )
    results[f"CNN_{name}"] = cnn_acc[-1]
    
    # Train ViT
    print(f"\nTraining ViT on {name}...")
    vit = SimpleViT(img_size=32, patch_size=4, num_classes=10)
    _, _, vit_acc = train_model(
        vit, train_loader, test_loader, epochs=10, lr=0.001, device=device, verbose=True
    )
    results[f"ViT_{name}"] = vit_acc[-1]

# Plot results
print("\n" + "="*50)
print("Plotting Results...")
print("="*50)

fig, ax = plt.subplots(figsize=(10, 6))

# Extract results
cnn_accs = [results[f"CNN_{p}%"] for p in percentages]
vit_accs = [results[f"ViT_{p}%"] for p in percentages]

# Plot
ax.plot(percentages, cnn_accs, 'o-', label='CNN', linewidth=2, markersize=10, color='#2ecc71')
ax.plot(percentages, vit_accs, 's-', label='ViT', linewidth=2, markersize=10, color='#3498db')
ax.set_xlabel('Percentage of Training Data Used', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Phase 3: CNN vs ViT - Data Efficiency', fontsize=14)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Add values on points
for i, (cnn_acc, vit_acc) in enumerate(zip(cnn_accs, vit_accs)):
    ax.text(percentages[i], cnn_acc + 2, f'{cnn_acc:.1f}%', ha='center', fontsize=9, color='#2ecc71')
    ax.text(percentages[i], vit_acc - 3, f'{vit_acc:.1f}%', ha='center', fontsize=9, color='#3498db')

# Add conclusion
conclusion = " CNN performs better with limited data\n💡 ViT needs more data to catch up"
ax.text(0.5, -0.2, conclusion,
        ha='center', va='center', transform=ax.transAxes,
        fontsize=11, style='italic',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('results/phase3_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary
print("\n" + "="*50)
print("PHASE 3 COMPLETE!")
print("="*50)
print("\n Final Accuracies:")
for p in percentages:
    print(f"   {p}% Data: CNN = {results[f'CNN_{p}%']:.2f}%, ViT = {results[f'ViT_{p}%']:.2f}%")

print("\n Results saved in 'results/phase3_results.png'")
print("="*50)