import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import TileDataset
from pathlib import Path

BASE_DIR = r'C:\Users\aljas\Desktop\diffusion_satellite_images\diffusion-minimal'

def compute_and_save_dataset_statistics():
    """
    Compute the mean, std, and histograms of the dataset on GPU and save them.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the dataset
    dataset = TileDataset(data_dir=f'{BASE_DIR}/data/tiles', image_size=64)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    histograms = torch.zeros(3, 256, device=device)  # RGB channels, 256 bins
    
    total_samples = 0

    for batch in tqdm(loader, desc='Computing dataset statistics'):
        if isinstance(batch, (list, tuple)):
            data = batch[0]  # Assuming batch[0] contains the images
        else:
            data = batch

        # Rescale to [0, 1] and move data to GPU
        data = ((data + 1) / 2).to(device)  # Scale from [-1, 1] to [0, 1]
        total_samples += data.size(0)

        # Compute mean and standard deviation
        mean += data.mean(dim=(0, 2, 3)) * data.size(0)
        std += data.std(dim=(0, 2, 3)) * data.size(0)

        # Compute histograms for each channel
        data_scaled = (data * 255).to(torch.uint8)  # Scale to [0, 255]
        for c in range(3):  # RGB channels
            histograms[c] += torch.histc(data_scaled[:, c, :, :].float(), bins=256, min=0, max=255)

    # Finalize the mean, std, and normalize the histograms
    mean /= total_samples
    std /= total_samples
    histograms /= total_samples  # Normalize histograms

    # Save statistics
    stats_path = Path(BASE_DIR, 'dataset_stats.pth')
    torch.save({'mean': mean, 'std': std, 'histograms': histograms}, stats_path)

    print(f'[INFO] Dataset statistics saved at {stats_path}')
    print(f'[DEBUG] Mean: {mean}, Std: {std}')
    print(f'[DEBUG] Histograms: {histograms}')

if __name__ == '__main__':
    compute_and_save_dataset_statistics()
