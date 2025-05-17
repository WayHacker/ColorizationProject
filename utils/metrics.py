import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import lab2rgb

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return psnr(img1, img2)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, multichannel=True)

def lab_to_rgb(L, ab):
    """Convert L and ab channels to RGB image"""
    L = L * 100.0
    ab = ab * 110.0
    
    Lab = np.concatenate([L, ab], axis=2)
    rgb_imgs = []
    
    for i in range(Lab.shape[0]):
        img_lab = Lab[i]
        img_rgb = lab2rgb(img_lab)
        rgb_imgs.append(img_rgb)
    
    return np.array(rgb_imgs)

def evaluate_model(model, dataloader, device):
    """Evaluate model performance using PSNR and SSIM metrics"""
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for L, ab_true in dataloader:
            L = L.to(device)
            ab_true = ab_true.to(device)
            
            # Forward pass
            ab_pred = model(L)
            
            # Convert to numpy for evaluation
            L_np = L.cpu().numpy()
            ab_true_np = ab_true.cpu().numpy()
            ab_pred_np = ab_pred.cpu().numpy()
            
            # Reshape for lab2rgb
            L_np = L_np.transpose(0, 2, 3, 1)  # [B, H, W, 1]
            ab_true_np = ab_true_np.transpose(0, 2, 3, 1)  # [B, H, W, 2]
            ab_pred_np = ab_pred_np.transpose(0, 2, 3, 1)  # [B, H, W, 2]
            
            # Convert to RGB
            rgb_true = lab_to_rgb(L_np, ab_true_np)
            rgb_pred = lab_to_rgb(L_np, ab_pred_np)
            
            # Calculate metrics
            for i in range(L.size(0)):
                psnr_values.append(calculate_psnr(rgb_true[i], rgb_pred[i]))
                ssim_values.append(calculate_ssim(rgb_true[i], rgb_pred[i]))
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }
