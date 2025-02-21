from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import filters
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms

# This implementation is based on open source resources, provided by Leon Pielage


class MNISTDataset(datasets.MNIST):

    def __init__(self, root='./data', image_size=28, train=True, download=True, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # transforms.Normalize((0,), (1,)),
            ])

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        return img * 2 - 1, label


class CIFAR10Dataset(datasets.CIFAR10):

    def __init__(self, root='./data', image_size=32, train=True, download=True, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # transforms.Normalize((0,), (1,)),
            ])

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        return img * 2 - 1, label


class TileDataset(Dataset):
    
    def __init__(self, data_dir, image_size=64, transform=None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        # Load .tiff files
        self.image_paths = list(self.data_dir.glob('*.tiff'))
        
        # Default transformations: resizing, converting to tensor, normalizing to [-1, 1] range
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1] for RGB channels
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open the .tiff file as RGB
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, 0  # Return a dummy label (e.g., 0)

