"""
Simple CNN with Adaptive Pooling - Works for ANY image size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN with adaptive pooling to handle different input sizes.
    This works for 16x16 (synthetic), 28x28 (MNIST), and 32x32 (CIFAR-10)
    """
    
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Adaptive pooling - makes ANY input size work!
        # Output will always be 4x4 regardless of input
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers - now we KNOW the size is 64 * 4 * 4
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Conv layers with ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Adaptive pooling to fixed size (4x4)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Quick test
if __name__ == "__main__":
    # Test with different image sizes
    test_sizes = [(1, 16, 16), (1, 28, 28), (3, 32, 32)]
    for channels, h, w in test_sizes:
        model = SimpleCNN(input_channels=channels, num_classes=10)
        test_input = torch.randn(1, channels, h, w)
        output = model(test_input)
        print(f"Input: {channels}x{h}x{w} -> Output shape: {output.shape}")