import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt

class ColorizationDataset(Dataset):
    def __init__(self, image_paths, split='train', img_size=256):
        self.image_paths = image_paths
        self.split = split
        
        # Define transformations
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        # Convert to LAB color space
        img_lab = rgb2lab(img.permute(1, 2, 0).numpy())
        
        # Normalize L channel to [0, 1] and ab channels to [-1, 1]
        L = img_lab[:, :, 0] / 100.0
        ab = img_lab[:, :, 1:] / 110.0
        
        return torch.from_numpy(L).unsqueeze(0).float(), torch.from_numpy(ab).permute(2, 0, 1).float()

def get_data_loaders(data_dir, batch_size=16, img_size=256):
    # Load dataset (e.g., CIFAR-10 for simplicity)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # For this example, we'll use CIFAR-10
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    # Extract image paths
    train_paths = [os.path.join(data_dir, 'cifar10', 'train', f"{i}.png") for i in range(len(train_dataset))]
    
    # Save images if they don't exist
    if not os.path.exists(os.path.join(data_dir, 'cifar10', 'train')):
        os.makedirs(os.path.join(data_dir, 'cifar10', 'train'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'cifar10', 'test'), exist_ok=True)
        
        for i, (img, _) in enumerate(train_dataset):
            img.save(os.path.join(data_dir, 'cifar10', 'train', f"{i}.png"))
        
        for i, (img, _) in enumerate(test_dataset):
            img.save(os.path.join(data_dir, 'cifar10', 'test', f"{i}.png"))
    
    # Split train into train and validation
    train_size = int(0.8 * len(train_paths))
    val_size = len(train_paths) - train_size
    train_paths, val_paths = torch.utils.data.random_split(train_paths, [train_size, val_size])
    
    test_paths = [os.path.join(data_dir, 'cifar10', 'test', f"{i}.png") for i in range(len(test_dataset))]
    
    # Create datasets
    train_dataset = ColorizationDataset(train_paths, split='train', img_size=img_size)
    val_dataset = ColorizationDataset(val_paths, split='val', img_size=img_size)
    test_dataset = ColorizationDataset(test_paths, split='test', img_size=img_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

def analyze_dataset(data_loader, num_samples=5):
    """Analyze and visualize sample images from the dataset"""
    # Get a batch of images
    L_batch, ab_batch = next(iter(data_loader))
    
    # Create a figure for visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        if i >= len(L_batch):
            break
            
        # Get L and ab channels
        L = L_batch[i].squeeze().numpy()
        ab = ab_batch[i].permute(1, 2, 0).numpy() * 110.0
        
        # Reconstruct color image
        lab_img = np.zeros((L.shape[0], L.shape[1], 3))
        lab_img[:, :, 0] = L * 100.0
        lab_img[:, :, 1:] = ab
        rgb_img = lab2rgb(lab_img)
        
        # Display grayscale image
        axes[i, 0].imshow(L, cmap='gray')
        axes[i, 0].set_title('Grayscale (L channel)')
        axes[i, 0].axis('off')
        
        # Display ab channels
        ab_display = np.zeros((ab.shape[0], ab.shape[1], 3))
        ab_display[:, :, 0] = (ab[:, :, 0] + 110) / 220  # Normalize for display
        ab_display[:, :, 1] = (ab[:, :, 1] + 110) / 220
        axes[i, 1].imshow(ab_display)
        axes[i, 1].set_title('Chrominance (ab channels)')
        axes[i, 1].axis('off')
        
        # Display color image
        axes[i, 2].imshow(rgb_img)
        axes[i, 2].set_title('Color (RGB)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    plt.close()
    
    # Analyze color distribution
    print("Analyzing color distribution...")
    L_values = []
    a_values = []
    b_values = []
    
    for L_batch, ab_batch in data_loader:
        for i in range(len(L_batch)):
            L = L_batch[i].squeeze().numpy().flatten() * 100.0
            a = ab_batch[i][0].numpy().flatten() * 110.0
            b = ab_batch[i][1].numpy().flatten() * 110.0
            
            L_values.extend(L)
            a_values.extend(a)
            b_values.extend(b)
            
        # Limit the number of samples for analysis
        if len(L_values) > 100000:
            break
    
    # Plot histograms
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
    plt.savefig('color_distribution.png')
    plt.close()
    
    # Print statistics
    print(f"L channel - Min: {min(L_values):.2f}, Max: {max(L_values):.2f}, Mean: {np.mean(L_values):.2f}, Std: {np.std(L_values):.2f}")
    print(f"a channel - Min: {min(a_values):.2f}, Max: {max(a_values):.2f}, Mean: {np.mean(a_values):.2f}, Std: {np.std(a_values):.2f}")
    print(f"b channel - Min: {min(b_values):.2f}, Max: {max(b_values):.2f}, Mean: {np.mean(b_values):.2f}, Std: {np.std(b_values):.2f}")
    
    return {
        'L_stats': {
            'min': min(L_values),
            'max': max(L_values),
            'mean': np.mean(L_values),
            'std': np.std(L_values)
        },
        'a_stats': {
            'min': min(a_values),
            'max': max(a_values),
            'mean': np.mean(a_values),
            'std': np.std(a_values)
        },
        'b_stats': {
            'min': min(b_values),
            'max': max(b_values),
            'mean': np.mean(b_values),
            'std': np.std(b_values)
        }
    }
