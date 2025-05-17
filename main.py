import os
import argparse
import torch
from utils.image_analysis import analyze_image_dataset
from train import train_model
from evaluate import evaluate
from colorize import colorize_image

def main(args):
    # Create necessary directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Analyze dataset if requested
    if args.analyze:
        print("Step 1: Analyzing dataset...")
        analyze_image_dataset(
            args.image_dir if args.image_dir else args.data_dir,
            num_samples=args.num_samples,
            save_dir=os.path.join(args.results_dir, 'analysis')
        )
    
    # Step 2: Train model if requested
    if args.train:
        print("Step 2: Training model...")
        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            epochs=args.epochs,
            save_freq=args.save_freq,
            analyze=False  # Already analyzed if requested
        )
        model, metrics = train_model(train_args)
        
        # Save best model path for evaluation
        best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    else:
        # Use provided model path
        best_model_path = args.model_path
    
    # Step 3: Evaluate model if requested
    if args.evaluate:
        print("Step 3: Evaluating model...")
        eval_args = argparse.Namespace(
            data_dir=args.data_dir,
            model_path=best_model_path,
            save_dir=os.path.join(args.results_dir, 'evaluation'),
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_samples=args.num_samples
        )
        metrics = evaluate(eval_args)
    
    # Step 4: Colorize sample images if requested
    if args.colorize and args.image_path:
        print("Step 4: Colorizing sample image...")
        colorize_args = argparse.Namespace(
            image_path=args.image_path,
            model_path=best_model_path,
            output_path=os.path.join(args.results_dir, 'colorized_sample.png'),
            img_size=args.img_size
        )
        colorize_image(colorize_args)
    
    print("All requested tasks completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Colorization Project')
    
    # General arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    
    # Task selection
    parser.add_argument('--analyze', action='store_true', help='Perform dataset analysis')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--colorize', action='store_true', help='Colorize a sample image')
    
    # Analysis arguments
    parser.add_argument('--image_dir', type=str, help='Directory containing images for analysis')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for analysis/visualization')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency of saving checkpoints')
    
    # Evaluation/Colorization arguments
    parser.add_argument('--model_path', type=str, help='Path to trained model (required if not training)')
    parser.add_argument('--image_path', type=str, help='Path to image for colorization')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.train and args.evaluate and not args.model_path:
        parser.error("--model_path is required when evaluating without training")
    
    if args.colorize and not args.image_path:
        parser.error("--image_path is required for colorization")
    
    main(args)