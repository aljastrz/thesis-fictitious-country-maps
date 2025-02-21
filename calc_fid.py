import torch
from pathlib import Path
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

# Paths
BASE_DIR = Path(r"C:\Users\aljas\Desktop\diffusion_satellite_images\diffusion-minimal")
REAL_TILES_DIR = BASE_DIR / "data" / "tiles_png"
STATS_PATH = BASE_DIR / "precomputed_real_stats.pth"
SINGLE_TILE_PATH = BASE_DIR / "generated_images" / "tiles" / "tile_38.png"

# Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_fid_for_single_tile(single_tile_path, device):
    """Calculate FID score for a single generated tile."""
    print(f"[INFO] Computing FID for {single_tile_path}")

    # Load InceptionV3 model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # Load or compute real dataset statistics
    if STATS_PATH.exists():
        print("[INFO] Loading precomputed real dataset statistics...")
        stats = torch.load(STATS_PATH)
        m1, s1 = stats['m1'], stats['s1']
    else:
        print("[INFO] Computing real dataset statistics (This will take time once)...")
        m1, s1 = compute_statistics_of_path(str(REAL_TILES_DIR), model, batch_size=50, device=device, dims=dims)
        torch.save({'m1': m1, 's1': s1}, STATS_PATH)

    # Compute statistics for the single tile
    m2, s2 = compute_statistics_of_path(str(single_tile_path.parent), model, batch_size=1, device=device, dims=dims)

    # **Fix Empty Covariance Matrix for Single Tile**
    if s2.shape == ():
        print("[WARNING] s2 is empty! Reshaping it to match s1...")
        s2 = torch.eye(dims, device=device) * 1e-6  # Small identity matrix as fallback

    # Ensure both covariance matrices match dimensions
    if m1.shape != m2.shape or s1.shape != s2.shape:
        print(f"[ERROR] Dimension mismatch: Real ({m1.shape}, {s1.shape}) vs Tile ({m2.shape}, {s2.shape})")
        return float("inf")  # Return a large FID value to indicate failure

    # Compute FID score
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

if __name__ == '__main__':
    fid_score = calculate_fid_for_single_tile(SINGLE_TILE_PATH, device)
    print(f"\n[RESULT] FID Score for {SINGLE_TILE_PATH.name}: {fid_score:.4f}")


