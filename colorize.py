import os
import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb

from models.unet import UNet

def colorize_image(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    img = Image.open(args.image_path).convert('RGB')
    
    # Resize image
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img)
    
    # Convert to LAB
    img_lab = rgb2lab(img_tensor.permute(1, 2, 0).numpy())
    L = img_lab[:, :, 0] / 100.0
    
    # Convert to tensor
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Colorize
    with torch.no_grad():
        ab_pred = model(L_tensor)
    
    # Convert back to numpy
    ab_pred = ab_pred.cpu().squeeze().permute(1, 2, 0).numpy() * 110.0
    
    # Combine L and ab channels
    L_np = L * 100.0
    lab_pred = np.zeros((L_np.shape[0], L_np.shape[1], 3))
    lab_pred[:, :, 0] = L_np
    lab_pred[:, :, 1:] = ab_pred
    
    # Convert to RGB
    rgb_pred = lab2rgb(lab_pred)
    
    # Save result
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(L_np, cmap='gray')
    plt.title('Grayscale Input')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_pred)
    plt.title('Colorized Output')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output_path)
    plt.close()
    
    print(f"Colorized image saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Colorize a grayscale image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to grayscale image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_path', type=str, default='colorized.png', help='Path to save colorized image')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for colorization')
    
    args = parser.parse_args()
    
    # Colorize image
    colorize_image(args)
