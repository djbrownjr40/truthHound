from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_data_loaders(data_dir, batch_size=32):
    """
    This function loads the training and validation datasets with appropriate transformations.
    
    Args:
        data_dir (str): Path to the root directory of the dataset (should contain 'train' and 'val' subfolders).
        batch_size (int): Number of samples per batch.
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation datasets.
    """
    # Define common transformations for both train and val data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize all images to 128x128
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])

    # Load training dataset
    train_data = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)

    # Load validation dataset
    val_data = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

    # Create DataLoader for batching
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)  # Don't shuffle validation data

    return train_loader, val_loader


# Example usage
data_dir = 'C:/Users/d.Brown/Desktop/d.brown/portfolio0/data'
train_loader, val_loader = get_data_loaders(data_dir, batch_size=32)
