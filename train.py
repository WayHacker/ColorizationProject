import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from data.dataset import get_data_loaders, analyze_dataset
from models.unet import UNet
from utils.metrics import evaluate_model
from utils.visualization import visualize_results

def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Analyze dataset
    if args.analyze:
        print("Analyzing dataset...")
        stats = analyze_dataset(train_loader, num_samples=5)
        print("Dataset analysis complete.")
    
    # Create model
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    
    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for L, ab in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            L, ab = L.to(device), ab.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(L)
            loss = criterion(outputs, ab)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for L, ab in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                L, ab = L.to(device), ab.to(device)
                
                # Forward pass
                outputs = model(L)
                loss = criterion(outputs, ab)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'))
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    metrics = evaluate_model(model, test_loader, device)
    print(f"Test PSNR: {metrics['psnr']:.4f}, Test SSIM: {metrics['ssim']:.4f}")
    
    # Visualize results
    print("Generating visualization...")
    visualize_results(model, test_loader, device, num_samples=5, save_dir=args.save_dir)
    
    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train colorization model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency of saving checkpoints')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset before training')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Train model
    model, metrics = train_model(args)
