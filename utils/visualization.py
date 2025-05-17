import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb
import os

def visualize_results(model, dataloader, device, num_samples=5, save_dir='results'):
    """Visualize colorization results"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    
    # Get a batch of images
    L_batch, ab_batch = next(iter(dataloader))
    L_batch = L_batch.to(device)
    
    # Generate predictions
    with torch.no_grad():
        ab_pred = model(L_batch)
    
    # Convert to numpy
    L_np = L_batch.cpu().numpy()
    ab_true_np = ab_batch.cpu().numpy()
    ab_pred_np = ab_pred.cpu().numpy()
    
    # Create a figure for visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(min(num_samples, len(L_batch))):
        # Get L and ab channels
        L = L_np[i].transpose(1, 2, 0) * 100.0  # [H, W, 1]
        ab_true = ab_true_np[i].transpose(1, 2, 0) * 110.0  # [H, W, 2]
        ab_pred = ab_pred_np[i].transpose(1, 2, 0) * 110.0  # [H, W, 2]
        
        # Reconstruct color images
        lab_true = np.concatenate([L, ab_true], axis=2)
        lab_pred = np.concatenate([L, ab_pred], axis=2)
        
        rgb_true = lab2rgb(lab_true)
        rgb_pred = lab2rgb(lab_pred)
        
        # Display grayscale image
        axes[i, 0].imshow(L.squeeze(), cmap='gray')
        axes[i, 0].set_title('Grayscale Input')
        axes[i, 0].axis('off')
        
        # Display ground truth color image
        axes[i, 1].imshow(rgb_true)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Display predicted color image
        axes[i, 2].imshow(rgb_pred)
        axes[i, 2].set_title('Predicted')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/colorization_results.png')
    plt.close()
    
    return fig
