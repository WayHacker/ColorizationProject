import os
import torch
import argparse
from tqdm import tqdm

from data.dataset import get_data_loaders
from models.unet import UNet
from utils.metrics import evaluate_model
from utils.visualization import visualize_results

def evaluate(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    _, _, test_loader = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Create model and load weights
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    print(f"Test PSNR: {metrics['psnr']:.4f}, Test SSIM: {metrics['ssim']:.4f}")
    
    # Visualize results
    print("Generating visualization...")
    visualize_results(model, test_loader, device, num_samples=args.num_samples, save_dir=args.save_dir)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate colorization model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory with dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for evaluation')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Evaluate model
    metrics = evaluate(args)
