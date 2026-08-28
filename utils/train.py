"""
Training utilities for CNN vs ViT experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu', verbose=True):
    """
    Generic training function for both CNN and ViT
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        verbose: Print progress
    
    Returns:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_accs: List of test accuracies per epoch
    """
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        if verbose:
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        else:
            progress_bar = train_loader
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if verbose and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(total/len(train_loader)):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        train_acc = 100. * correct / total
        train_accs.append(train_acc)
        train_losses.append(total_loss / len(train_loader))
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        test_accs.append(test_acc)
        
        if verbose:
            print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"Training completed in {elapsed:.2f} seconds")
    
    return train_losses, train_accs, test_accs