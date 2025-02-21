from pathlib import Path
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from trainer import DiffusionTrainer
from configs import DiffusionConfigs
import matplotlib.pyplot as plt
from torchvision.models import inception_v3
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths
import os

# Define base directory
BASE_DIR = r'C:\Users\...'
initial_path = r"C:\Users\...\tile_2023-03-31_subregion_3_12_RGB_15_0.tiff"

# Load dataset statistics
stats_path = Path(BASE_DIR, 'dataset_stats.pth')
if stats_path.exists():
    dataset_stats = torch.load(stats_path)
    histograms_ref = dataset_stats['histograms'].to('cuda')  # Ensure reference histograms are on GPU
    print(f"[INFO] Loaded dataset histograms")
else:
    raise FileNotFoundError(f"Dataset statistics file not found at {stats_path}")

def load_tile(file_path, device):
    """Load an image tile and normalize it to [-1, 1]."""
    img = Image.open(file_path).convert("RGB")
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255  # Convert to tensor in range [0, 1]
    img_tensor = img_tensor * 2 - 1  # Scale to [-1, 1]
    return img_tensor.unsqueeze(0).to(device)  # Add batch dimension

def calculate_histograms(tile):
    """Calculate histograms for a tile for RGB channels (on GPU)."""
    tile = (tile * 255).clamp(0, 255).to(torch.uint8).to('cuda')  # Scale to [0, 255]
    histograms = torch.zeros(3, 256, device='cuda')
    for c in range(3):
        histograms[c] = torch.histc(tile[c].float(), bins=256, min=0, max=255)
    return histograms

def plot_tile_and_histogram(tile, histograms, title):
    """Display the tile and its histograms for visual validation."""
    tile_image = (tile.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Display tile
    axes[0].imshow(tile_image)
    axes[0].set_title(f"{title}")
    axes[0].axis('off')

    # Display histograms
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        axes[i + 1].bar(range(256), histograms[i].cpu().numpy(), color=color, alpha=0.7)
        axes[i + 1].set_title(f"{color.capitalize()} Histogram")
        axes[i + 1].set_xlim(0, 255)

    plt.show()

def is_valid_tile(tile, histograms_ref, red_mean_range=(35, 220), green_mean_range=(20, 190), blue_mean_range=(5, 80)):
    """
    Check if a tile is valid based on histogram statistics for RGB channels and display the results.

    Parameters:
    - tile: The tile tensor to validate.
    - histograms_ref: Reference histograms from the dataset.
    - red_mean_range, green_mean_range, blue_mean_range: Valid mean ranges for each channel.

    Returns:
    - bool: True if the tile is valid, False otherwise.
    """
    histograms_tile = calculate_histograms(tile)

    # Calculate mean for each channel
    means = []
    for i, hist in enumerate(histograms_tile):
        pixel_values = torch.arange(256, device='cuda', dtype=torch.float32)
        mean = torch.sum(hist * pixel_values) / torch.sum(hist)
        means.append(mean)

    red_mean, green_mean, blue_mean = means

    # Validate mean ranges
    if not (red_mean_range[0] <= red_mean <= red_mean_range[1]):
        print(f"[DEBUG] Tile rejected: Red mean {red_mean.item():.2f} out of range {red_mean_range}")
        # plot_tile_and_histogram(tile, histograms_tile, "Invalid Tile (Red)")
        return False

    if not (green_mean_range[0] <= green_mean <= green_mean_range[1]):
        print(f"[DEBUG] Tile rejected: Green mean {green_mean.item():.2f} out of range {green_mean_range}")
        # plot_tile_and_histogram(tile, histograms_tile, "Invalid Tile (Green)")
        return False

    if not (blue_mean_range[0] <= blue_mean <= blue_mean_range[1]):
        print(f"[DEBUG] Tile rejected: Blue mean {blue_mean.item():.2f} out of range {blue_mean_range}")
        # plot_tile_and_histogram(tile, histograms_tile, "Invalid Tile (Blue)")
        return False

    
    # plot_tile_and_histogram(tile, histograms_tile, "Valid Tile")
    print(f"[DEBUG] Tile accepted. Red Mean: {red_mean.item():.2f}, Green Mean: {green_mean.item():.2f}, Blue Mean: {blue_mean.item():.2f}")
    return True

def add_noise_to_overlap(overlap_region, diffusion_model, t_step=0):
    """Add noise to the overlap region using the diffusion model."""
    noise = diffusion_model.q_sample(overlap_region, t=torch.tensor([t_step], device=overlap_region.device))
    return noise


def load_initial_tile(tile_path, device):
    """Load the initial tile from a specified path."""
    tile = Image.open(tile_path).convert("RGB")
    tile = np.array(tile).transpose(2, 0, 1)  # Convert to CHW format
    tile_tensor = torch.tensor(tile, dtype=torch.float32).div(255).mul(2).sub(1)  # Normalize to [-1, 1]
    return tile_tensor.unsqueeze(0).to(device)  # Add batch dimension

def generate_seamless_map(trainer: DiffusionTrainer, grid_size: tuple, output_path: str, overlap: int):
    """
    Generate a coherent map with realistic border matching.

    :param trainer: Instance of DiffusionTrainer.
    :param grid_size: Tuple (rows, cols) specifying the grid dimensions.
    :param output_path: Path to save the coherent map.
    :param overlap: Overlap (in pixels) between adjacent tiles.
    """
    rows, cols = grid_size
    tile_size = trainer.configs.image_size  # Size of each tile
    n_channels = trainer.configs.image_channels

    # Adjust map dimensions to account for overlap
    map_height = rows * tile_size - (rows - 1) * overlap
    map_width = cols * tile_size - (cols - 1) * overlap
    full_map = torch.zeros((n_channels, map_height, map_width), device=trainer.configs.device)

    print(f"[DEBUG] Grid size: {rows}x{cols}, Tile size: {tile_size}, Overlap: {overlap}")

    generated_images_dir = Path(BASE_DIR, "generated_images")
    generated_images_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for row in range(rows):
            for col in range(cols):
                # Compute the position of the current tile
                start_y = row * (tile_size - overlap)
                start_x = col * (tile_size - overlap)
                end_y = start_y + tile_size
                end_x = start_x + tile_size

                # Initialize previous overlap areas
                top_previous, left_previous = None, None

                if row > 0:
                    top_tile_start = start_y - overlap
                    top_previous = full_map[:, top_tile_start:start_y, start_x:end_x].clone()

                if col > 0:
                    left_tile_start = start_x - overlap
                    left_previous = full_map[:, start_y:end_y, left_tile_start:start_x].clone()

                print(f"[DEBUG] Generating tile at ({row}, {col})")

                # Generate the tile
                valid_tile = False
                attempts = 0
                max_attempts = 20

                while not valid_tile and attempts < max_attempts:
                    # Generate a new tile with better overlap blending
                    new_tile = trainer.sample_batch(
                        path=None,
                        n_samples=1,
                        latent_batch=None,
                        latent_without_noise=False,
                        left_previous=left_previous,
                        top_previous=top_previous,
                        save_result=False,
                        seed=None,
                    ).squeeze(0)

                    # Rescale and validate the tile
                    new_tile_rescaled = (new_tile + 1) / 2
                    valid_tile = is_valid_tile(new_tile_rescaled, histograms_ref)
                    attempts += 1

                    if not valid_tile:
                        print(f"[DEBUG] Invalid tile at ({row}, {col}). Regenerating... (Attempt {attempts}/{max_attempts})")

                if not valid_tile:
                    print(f"[WARNING] Failed to generate valid tile at ({row}, {col}) after {max_attempts} attempts.")

                # **Intelligent Blending of Overlaps**
                if row > 0:
                    mask = torch.linspace(0, 1, overlap, device=trainer.configs.device).view(1, overlap, 1)
                    blend_top = new_tile[:, :overlap, :] * mask + full_map[:, start_y:start_y + overlap, start_x:end_x] * (1 - mask)
                    full_map[:, start_y:start_y + overlap, start_x:end_x] = blend_top

                if col > 0:
                    mask = torch.linspace(0, 1, overlap, device=trainer.configs.device).view(1, 1, overlap)
                    blend_left = new_tile[:, :, :overlap] * mask + full_map[:, start_y:end_y, start_x:start_x + overlap] * (1 - mask)
                    full_map[:, start_y:end_y, start_x:start_x + overlap] = blend_left

                # Save individual tiles for FID calculation
                tile_path = generated_images_dir / f"tile_{row}_{col}.png"
                save_image(new_tile_rescaled, tile_path)

                # **Ensure tile placement without direct copying**
                full_map[:, start_y:end_y, start_x:end_x] = new_tile

    # Rescale the map to [0, 1] for saving
    full_map = (full_map + 1) / 2
    full_map = full_map.clamp(0, 1)

    save_image(full_map, output_path)
    print(f"[DEBUG] Seamless map saved at {output_path}")

    # Compute FID
    dataset_dir = Path(BASE_DIR, "data", "tiles")


# Example usage
if __name__ == "__main__":
    # Define checkpoint directory
    checkpoint_dir = Path(BASE_DIR) / "projects" / "Satellite_Image_Test" / "checkpoints"

    # List and sort checkpoints
    checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)

    if checkpoints:
        print(f"[INFO] Found {len(checkpoints)} checkpoints in {checkpoint_dir}")
        for ckpt in checkpoints:
            print(f"  - {ckpt.name}, last modified: {ckpt.stat().st_mtime}")

        # Select the highest-numbered checkpoint (not just the most recent)
        latest_checkpoint = max(
            checkpoints, key=lambda ckpt: int(ckpt.stem.split('-')[-1]) if '-' in ckpt.stem else -1
        )

        print(f"[INFO] Loading latest checkpoint: {latest_checkpoint}")

        # Initialize configurations BEFORE trainer initialization
        configs = DiffusionConfigs(
            name='Satellite_Image_Test',
            dim=32,
            dim_mults=(1, 2),
            image_channels=3,
            image_size=64,
            batch_size=64,
            epochs=1000,
            seed=42,
            save_interval=50,
            lr_scheduler_patience=50,
        )

        # Explicitly set the correct checkpoint path
        configs._resume_file_name = latest_checkpoint.name
        configs._save_file_name = latest_checkpoint.name  # Ensure consistency
        configs.start_epoch = int(latest_checkpoint.stem.split('-')[-1])  # Set correct epoch
        
        # Load the checkpoint BEFORE initializing the trainer
        configs.load_checkpoint(path=str(latest_checkpoint))  
        print(f"[INFO] Successfully loaded checkpoint from {latest_checkpoint}")

    else:
        print("[WARNING] No checkpoints found! Make sure the training script saved them correctly.")
        exit(1)  # Exit to prevent running without a checkpoint

    # Now initialize the trainer AFTER loading the correct checkpoint
    trainer = DiffusionTrainer(configs)

    # No need to call trainer.load_checkpoint(), since it does not exist
    print(f"[INFO] Model is now using checkpoint from epoch {configs.start_epoch}")

    # Generate a coherent map
    generate_seamless_map(
        trainer=trainer,
        grid_size=(2, 2),  # Grid size: 2x2
        output_path=f"{BASE_DIR}/seamless_map_new.png",
        overlap=8  # Overlap in pixels
    )