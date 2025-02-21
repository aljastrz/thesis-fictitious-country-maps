import torch
import matplotlib.pyplot as plt

# Load the checkpoint
checkpoint_path = r"C:\Users\...\dataset_stats.pth"
checkpoint = torch.load(checkpoint_path)

# Check and print the top-level keys in the checkpoint
print("\n[Checkpoint Keys]")
for key in checkpoint.keys():
    print(f"- {key}")

if "mean" in checkpoint and "std" in checkpoint:
    dataset_id = "dataset_statistics"
    print("\n[Dataset Statistics]")
    print(f"ID: {dataset_id}")

    # Move statistics to GPU
    mean = checkpoint["mean"].to("cuda")
    std = checkpoint["std"].to("cuda")
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    # If histograms exist, display them
    if "histograms" in checkpoint:
        histograms = checkpoint["histograms"].to("cuda")  # Move histograms to GPU

        # Visualize histograms
        channels = ["Red", "Green", "Blue"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, (channel, hist) in enumerate(zip(channels, histograms)):
            hist_cpu = hist.cpu().numpy()  # Move to CPU for plotting
            axes[i].bar(range(256), hist_cpu, color=channel.lower(), alpha=0.7)
            axes[i].set_title(f"{channel} Histogram")
            axes[i].set_xlim(0, 255)
            axes[i].set_xlabel("Pixel Intensity")
            axes[i].set_ylabel("Frequency")

            # Compute additional statistics for histograms on GPU
            pixel_values = torch.arange(256, device="cuda", dtype=torch.float32)
            mean_value = torch.sum(hist * pixel_values) / torch.sum(hist)
            variance = torch.sum(hist * (pixel_values - mean_value) ** 2) / torch.sum(hist)
            std_dev = variance.sqrt()

            print(f"\n[{channel} Histogram Statistics]")
            print(f"- Mean: {mean_value.item():.2f}")
            print(f"- Std Dev: {std_dev.item():.2f}")

        plt.tight_layout()
        plt.show()

else:
    print("\n[Dataset Statistics]")
    print("This checkpoint does not contain dataset statistics.")

print("\n[Configs]")
if "configs" in checkpoint:
    configs = checkpoint["configs"]
    for config_key, config_value in configs.items():
        print(f"{config_key}: {config_value}")
else:
    print("No 'configs' found in the checkpoint.")

# Analyze the model state dictionary
print("\n[Model State Dict]")
if "model_state_dict" in checkpoint:
    model_state_dict = checkpoint["model_state_dict"]
    for param_name, param_tensor in model_state_dict.items():
        print(f"Parameter: {param_name}, Shape: {param_tensor.shape}")
else:
    print("No 'model_state_dict' found in the checkpoint.")

# Analyze the optimizer state dictionary
print("\n[Optimizer State Dict]")
if "optimizer_state_dict" in checkpoint:
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    print(f"Keys in optimizer_state_dict: {list(optimizer_state_dict.keys())}")
else:
    print("No 'optimizer_state_dict' found in the checkpoint.")

# Analyze the learning rate scheduler state dictionary
print("\n[Learning Rate Scheduler State Dict]")
if "learning_rate_scheduler_state_dict" in checkpoint:
    lr_scheduler_state_dict = checkpoint["learning_rate_scheduler_state_dict"]
    print(f"Keys in lr_scheduler_state_dict: {list(lr_scheduler_state_dict.keys())}")
else:
    print("No 'learning_rate_scheduler_state_dict' found in the checkpoint.")

# Check if there are unexpected keys in the checkpoint
print("\n[Other Keys]")
known_keys = {"configs", "model_state_dict", "optimizer_state_dict", "learning_rate_scheduler_state_dict"}
unexpected_keys = set(checkpoint.keys()) - known_keys
if unexpected_keys:
    print("Unexpected keys found:", unexpected_keys)
else:
    print("No unexpected keys found.")