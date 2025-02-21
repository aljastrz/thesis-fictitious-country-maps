from pathlib import Path
from trainer import DiffusionTrainer
from configs import DiffusionConfigs

# Define base directory
BASE_DIR = r'C:\Users\aljas\Desktop\diffusion_satellite_images\diffusion-minimal'

def sample_from_model(output_path: str):
    """
    Samples images from the diffusion model using the latest checkpoint.

    :param output_path: Path to save the generated sample image.
    """
    # Load the configuration and trainer
    configs = DiffusionConfigs(
        name='Satellite_Image_Test',
        dim=32,                     # Increased dimensionality for better feature extraction
        dim_mults=(1, 2),           # Use a deeper network with more scale levels
        image_channels=3,           # Set to 3 for RGB
        image_size=64,              # Keep tile resolution the same
        batch_size=64,              # Reduce batch size to avoid memory issues
        epochs=1000,                # Fewer epochs if training is slow
        seed=42,                    # Reproducibility
        save_interval=50,           # Save checkpoints more frequently
        lr_scheduler_patience=50,   # Adjust learning rate sooner
    )
    configs.load_checkpoint(latest=True)  # Load the latest checkpoint
    trainer = DiffusionTrainer(configs)

    # Sample from the model
    print(f"[INFO] Sampling from the model. Output will be saved at {output_path}")
    trainer.sample(path=output_path)
    print(f"[INFO] Sampling complete. File saved to {output_path}")


if __name__ == "__main__":
    # Define the output path for the generated sample
    output_path = f"{BASE_DIR}/result_c.png"

    # Call the sampling function
    sample_from_model(output_path)
