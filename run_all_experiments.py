"""
CNN vs Vision Transformer: Complete Learning Lab
Run all experiments sequentially
"""

import os
import sys

# Create results folder if it doesn't exist
os.makedirs('results', exist_ok=True)

print("="*60)
print("CNN vs Vision Transformer: Complete Learning Lab")
print("="*60)
print("\nThis will run all experiments sequentially.")
print("Results and plots will be saved in the 'results' folder.")
print("\n Estimated Total Time: 30-40 minutes")
print("   - Phase 1: 2 minutes")
print("   - Phase 2: 10 minutes")
print("   - Phase 3: 20-25 minutes")
print("   - Phase 4: 5 minutes")
print("\nPress Enter to continue or Ctrl+C to cancel...")
input()

# PHASE 1: Synthetic Lines

print("\n" + "="*60)
print("Starting Phase 1: Synthetic Lines")
print("="*60)
print(" Testing learning speed on simple patterns")
print("  Estimated time: 2 minutes")
print("-"*60)

try:
    # Check if file exists
    if os.path.exists('practice-experiments/phase1_synthetic_lines.py'):
        exec(open('practice-experiments/phase1_synthetic_lines.py').read())
        print(" Phase 1 complete!")
    else:
        print(" File not found: practice-experiments/phase1_synthetic_lines.py")
        print("   Please create this file first.")
except Exception as e:
    print(f" Phase 1 failed: {e}")


# PHASE 2: Shuffled MNIST

print("\n" + "="*60)
print("Starting Phase 2: Shuffled MNIST")
print("="*60)
print(" Testing spatial understanding on shuffled digits")
print("  Estimated time: 10 minutes")
print("-"*60)

try:
    if os.path.exists('practice-experiments/phase2_shuffled_mnist.py'):
        exec(open('practice-experiments/phase2_shuffled_mnist.py').read())
        print(" Phase 2 complete!")
    else:
        print(" File not found: practice-experiments/phase2_shuffled_mnist.py")
        print("   Skipping Phase 2...")
except Exception as e:
    print(f" Phase 2 failed: {e}")

# PHASE 3: CIFAR-10 Data Subsets

print("\n" + "="*60)
print("Starting Phase 3: CIFAR-10 Data Subsets")
print("="*60)
print(" Testing data efficiency on 5%, 20%, 50%, 100% data")
print("  Estimated time: 20-25 minutes")
print("-"*60)

try:
    if os.path.exists('practice-experiments/phase3_data_efficiency.py'):
        exec(open('practice-experiments/phase3_data_efficiency.py').read())
        print(" Phase 3 complete!")
    else:
        print(" File not found: practice-experiments/phase3_data_efficiency.py")
        print("   Skipping Phase 3...")
except Exception as e:
    print(f" Phase 3 failed: {e}")


# PHASE 4: Visualization

print("\n" + "="*60)
print("Starting Phase 4: Model Visualization")
print("="*60)
print(" Visualizing CNN feature maps and ViT attention")
print("⏱  Estimated time: 5 minutes")
print("-"*60)

try:
    if os.path.exists('practice-experiments/phase4_visualization.py'):
        exec(open('practice-experiments/phase4_visualization.py').read())
        print(" Phase 4 complete!")
    else:
        print(" File not found: practice-experiments/phase4_visualization.py")
        print("   Skipping Phase 4...")
except Exception as e:
    print(f" Phase 4 failed: {e}")

# SUMMARY

print("\n" + "="*60)
print(" ALL EXPERIMENTS COMPLETE!")
print("="*60)
print("\n Results saved in the 'results' folder:")
print("   Phase 1: phase1_samples.png, phase1_comparison.png")
print("   Phase 2: phase2_samples.png, phase2_comparison.png")
print("   Phase 3: phase3_results.png")
print("   Phase 4: phase4_heatmaps.png")

print("\n Key Takeaways:")
print("   ✓ CNNs learn fast but break when structure is shuffled")
print("   ✓ ViTs are robust to spatial changes but need more data")
print("   ✓ Choose CNN for small data, ViT for large data")

print("\n Check the plots to see the differences!")
print("="*60)