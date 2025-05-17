import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
from torchvision import transforms
from tqdm import tqdm

def analyze_image_dataset(image_dir, num_samples=100, save_dir='analysis_results'):
    """Analyze a dataset of images for colorization task"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get list of image files
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly sample images
    if len(image_files) > num_samples:
        image_files = np.random.choice(image_files, num_samples, replace=False)
    
    # Initialize arrays for analysis
    L_values = []
    a_values = []
    b_values = []
    image_sizes = []
    
    # Process images
    for img_path in tqdm(image_files, desc="Analyzing images"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Record image size
            image_sizes.append(img.size)
            
            # Convert to numpy array
            img_np = np.array(img) / 255.0
            
            # Convert to LAB color space
            img_lab = rgb2lab(img_np)
            
            # Extract channels
            L = img_lab[:, :, 0].flatten()
            a = img_lab[:, :, 1].flatten()
            b = img_lab[:, :, 2].flatten()
            
            # Store values
            L_values.extend(L)
            a_values.extend(a)
            b_values.extend(b)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    L_values = np.array(L_values)
    a_values = np.array(a_values)
    b_values = np.array(b_values)
    
    # Analyze image sizes
    widths, heights = zip(*image_sizes)
    plt.figure(figsize=(10, 5))
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Image Dimensions')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.savefig(os.path.join(save_dir, 'image_dimensions.png'))
    plt.close()
    
    # Plot histograms of LAB values
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(L_values, bins=50)
    axes[0].set_title('L Channel Distribution')
    axes[0].set_xlabel('L Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(a_values, bins=50)
    axes[1].set_title('a Channel Distribution')
    axes[1].set_xlabel('a Value')
    axes[1].set_ylabel('Frequency')
    
    axes[2].hist(b_values, bins=50)
    axes[2].set_title('b Channel Distribution')
    axes[2].set_xlabel('b Value')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lab_distributions.png'))
    plt.close()
    
    # Plot 2D histogram of a and b channels
    plt.figure(figsize=(10, 8))
    plt.hist2d(a_values, b_values, bins=100, cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.title('Distribution of a and b Channels')
    plt.xlabel('a Channel')
    plt.ylabel('b Channel')
    plt.savefig(os.path.join(save_dir, 'ab_distribution_2d.png'))
    plt.close()
    
    # Visualize sample images in RGB and LAB
    sample_indices = np.random.choice(len(image_files), min(5, len(image_files)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        img_path = image_files[idx]
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img) / 255.0
        
        # Convert to LAB
        img_lab = rgb2lab(img_np)
        
        # Extract channels
        L = img_lab[:, :, 0]
        a = img_lab[:, :, 1]
        b = img_lab[:, :, 2]
        
        # Create grayscale image
        gray_img = np.stack([L/100.0] * 3, axis=-1)
        
        # Normalize a and b for visualization
        a_norm = (a + 128) / 255.0
        b_norm = (b + 128) / 255.0
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray_img, cmap='gray')
        axes[0, 1].set_title('Grayscale (L channel)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(L, cmap='gray')
        axes[0, 2].set_title('L Channel')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(a, cmap='RdBu_r', vmin=-128, vmax=127)
        axes[1, 0].set_title('a Channel')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(b, cmap='RdYlBu_r', vmin=-128, vmax=127)
        axes[1, 1].set_title('b Channel')
        axes[1, 1].axis('off')
        
        # Reconstruct from L only (grayscale)
        lab_gray = np.zeros_like(img_lab)
        lab_gray[:, :, 0] = L
        rgb_gray = lab2rgb(lab_gray)
        
        axes[1, 2].imshow(rgb_gray)
        axes[1, 2].set_title('Reconstructed Grayscale')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_analysis_{i}.png'))
        plt.close()
    
    # Print statistics
    print("Dataset Statistics:")
    print(f"Number of images analyzed: {len(image_files)}")
    print(f"Average image dimensions: {np.mean(widths):.1f} x {np.mean(heights):.1f} pixels")
    print("\nL channel statistics:")
    print(f"  Min: {np.min(L_values):.2f}, Max: {np.max(L_values):.2f}")
    print(f"  Mean: {np.mean(L_values):.2f}, Std: {np.std(L_values):.2f}")
    print("\na channel statistics:")
    print(f"  Min: {np.min(a_values):.2f}, Max: {np.max(a_values):.2f}")
    print(f"  Mean: {np.mean(a_values):.2f}, Std: {np.std(a_values):.2f}")
    print("\nb channel statistics:")
    print(f"  Min: {np.min(b_values):.2f}, Max: {np.max(b_values):.2f}")
    print(f"  Mean: {np.mean(b_values):.2f}, Std: {np.std(b_values):.2f}")
    
    # Save statistics to file
    with open(os.path.join(save_dir, 'dataset_statistics.txt'), 'w') as f:
        f.write("Dataset Statistics:\n")
        f.write(f"Number of images analyzed: {len(image_files)}\n")
        f.write(f"Average image dimensions: {np.mean(widths):.1f} x {np.mean(heights):.1f} pixels\n\n")
        f.write("L channel statistics:\n")
        f.write(f"  Min: {np.min(L_values):.2f}, Max: {np.max(L_values):.2f}\n")
        f.write(f"  Mean: {np.mean(L_values):.2f}, Std: {np.std(L_values):.2f}\n\n")
        f.write("a channel statistics:\n")
        f.write(f"  Min: {np.min(a_values):.2f}, Max: {np.max(a_values):.2f}\n")
        f.write(f"  Mean: {np.mean(a_values):.2f}, Std: {np.std(a_values):.2f}\n\n")
        f.write("b channel statistics:\n")
        f.write(f"  Min: {np.min(b_values):.2f}, Max: {np.max(b_values):.2f}\n")
        f.write(f"  Mean: {np.mean(b_values):.2f}, Std: {np.std(b_values):.2f}\n")
    
    return {
        'L_stats': {
            'min': np.min(L_values),
            'max': np.max(L_values),
            'mean': np.mean(L_values),
            'std': np.std(L_values)
        },
        'a_stats': {
            'min': np.min(a_values),
            'max': np.max(a_values),
            'mean': np.mean(a_values),
            'std': np.std(a_values)
        },
        'b_stats': {
            'min': np.min(b_values),
            'max': np.max(b_values),
            'mean': np.mean(b_values),
            'std': np.std(b_values)
        },
        'image_sizes': {
            'mean_width': np.mean(widths),
            'mean_height': np.mean(heights),
            'min_width': np.min(widths),
            'min_height': np.min(heights),
            'max_width': np.max(widths),
            'max_height': np.max(heights)
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze image dataset for colorization')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of images to analyze')
    parser.add_argument('--save_dir', type=str, default='analysis_results', help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    analyze_image_dataset(args.image_dir, args.num_samples, args.save_dir)
