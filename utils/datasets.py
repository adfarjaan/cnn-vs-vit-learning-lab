"""
Dataset utilities for CNN vs ViT experiments
Includes: Synthetic Lines, Shuffled MNIST, CIFAR-10 Subsets
"""

import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os

class SyntheticLinesDataset(Dataset):
    """Creates 16x16 images with vertical or horizontal lines"""
    
    def __init__(self, num_samples=2000, img_size=16):
        self.num_samples = num_samples
        self.img_size = img_size
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            # Blank image
            img = np.zeros((img_size, img_size), dtype=np.float32)
            
            # Label: 0 = vertical, 1 = horizontal
            label = 0 if i < num_samples // 2 else 1
            
            # Draw line with random position and thickness
            if label == 0:  # Vertical
                col = random.randint(4, img_size - 5)
                thickness = random.randint(1, 3)
                for c in range(col - thickness//2, col + thickness//2 + 1):
                    if 0 <= c < img_size:
                        img[:, c] = 1.0
            else:  # Horizontal
                row = random.randint(4, img_size - 5)
                thickness = random.randint(1, 3)
                for r in range(row - thickness//2, row + thickness//2 + 1):
                    if 0 <= r < img_size:
                        img[r, :] = 1.0
            
            # Add noise
            noise = np.random.normal(0, 0.05, (img_size, img_size))
            img = np.clip(img + noise, 0, 1)
            
            # Add channel dimension
            img = np.expand_dims(img, axis=0)
            
            self.images.append(img)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.images[idx]), torch.LongTensor([self.labels[idx]])[0]


class ShuffledMNIST(Dataset):
    """MNIST with quadrants randomly shuffled to test spatial understanding"""
    
    def __init__(self, train=True, shuffle=True, shuffle_prob=0.5):
        # Load normal MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download if not exists
        self.mnist = torchvision.datasets.MNIST(
            root='./data', train=train, download=True, transform=transform
        )
        self.shuffle = shuffle
        self.shuffle_prob = shuffle_prob
        
        # Pre-generate shuffle permutations for consistency
        self.permutations = []
        for _ in range(100):
            perm = random.sample([0, 1, 2, 3], 4)
            self.permutations.append(perm)
    
    def __len__(self):
        return len(self.mnist)
    
    def shuffle_quadrants(self, img):
        """Split 28x28 image into 4 quadrants and reshuffle"""
        img_np = img.squeeze().numpy()
        h, w = img_np.shape
        h_half, w_half = h // 2, w // 2
        
        # Extract quadrants
        quadrants = [
            img_np[:h_half, :w_half],      # Top-left (0)
            img_np[:h_half, w_half:],      # Top-right (1)
            img_np[h_half:, :w_half],      # Bottom-left (2)
            img_np[h_half:, w_half:]       # Bottom-right (3)
        ]
        
        # Random shuffle
        perm_idx = random.randint(0, len(self.permutations) - 1)
        perm = self.permutations[perm_idx]
        shuffled = [
            quadrants[perm[0]], quadrants[perm[1]],
            quadrants[perm[2]], quadrants[perm[3]]
        ]
        
        # Reassemble
        new_img = np.zeros_like(img_np)
        new_img[:h_half, :w_half] = shuffled[0]
        new_img[:h_half, w_half:] = shuffled[1]
        new_img[h_half:, :w_half] = shuffled[2]
        new_img[h_half:, w_half:] = shuffled[3]
        
        return torch.FloatTensor(new_img).unsqueeze(0)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        if self.shuffle and random.random() < self.shuffle_prob:
            img = self.shuffle_quadrants(img)
        
        return img, label


def create_cifar_subsets():
    """Create 5%, 20%, 50%, and 100% subsets of CIFAR-10"""
    
    # CIFAR-10 transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full CIFAR-10
    print("Loading CIFAR-10 dataset...")
    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Get indices for each class
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(full_train):
        class_indices[label].append(idx)
    
    # Create subsets
    subsets = {}
    for percentage in [5, 20, 50, 100]:
        if percentage == 100:
            indices = list(range(len(full_train)))
        else:
            indices = []
            for class_id, class_idx_list in class_indices.items():
                num_samples = int(len(class_idx_list) * percentage / 100)
                sampled = random.sample(class_idx_list, num_samples)
                indices.extend(sampled)
            random.shuffle(indices)
        
        subsets[f"{percentage}%"] = Subset(full_train, indices)
        print(f"  {percentage}% subset: {len(subsets[f'{percentage}%'])} images")
    
    return subsets, testset