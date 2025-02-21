import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

import main_yaml as my
from configs import DiffusionConfigs
from datasets import TileDataset  # Use your custom dataset for RGB tiles
from trainer import DiffusionTrainer
from misc import reset_seed, save_images_split

# This implementation is based on open source resources, provided by Leon Pielage

BASE_DIR = r'C:\Users\...'


def main():
    parser = argparse.ArgumentParser(description='Train a (conditional) diffusion model and sample from it')
    subparsers = parser.add_subparsers(dest='command')

    parser_diffusion = subparsers.add_parser('diffusion', help='Train a diffusion model')
    parser_diffusion.add_argument('--sample', action='store_true', help='Sample from the diffusion model')

    parser_samplecc = subparsers.add_parser('sampleuc', help='Sample from an unconditional model')
    parser_samplecc.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size (default: 8)')
    parser_samplecc.add_argument('-n', '--n-samples', type=int, default=8, help='Number of samples (default: 8)')

    parser_sampledata = subparsers.add_parser('sampledata', help='Sample directly from the dataset')
    parser_sampledata.add_argument('-n', '--n-samples', type=int, default=8, help='Number of samples to extract (default: 8)')

    args = parser.parse_args()
    if args.command == 'diffusion':
        train_diffusion(train_phase=(not args.sample))
    elif args.command == 'sampleuc':
        sample_unconditional(args)
    elif args.command == 'sampledata':
        sample_data(args)


def train_diffusion(train_phase: bool = True):
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

    if train_phase:
        configs.load_checkpoint()
        trainer = DiffusionTrainer(configs)

        # Load the full dataset
        full_data = TileDataset(data_dir=f'{BASE_DIR}/data/tiles', image_size=configs.image_size)

        # Train on the full dataset
        trainer.train(full_data, auto_val_size=0.1)
        configs.save()
    else:
        configs.load_checkpoint(latest=True)
        trainer = DiffusionTrainer(configs)
        trainer.sample(f'{BASE_DIR}/result.png')


def sample_unconditional(args):
    setup = {
        'n_samples': args.n_samples,
        'batch_size': args.batch_size,
        'overwrite': False,
        'output_dir': f'{BASE_DIR}/results',
        'configs': {
            'type': 'DiffusionConfigs',
            'params': {
                'name': 'Satellite_Image_Test',
            },
        },
    }
    my.sample_unconditional(setup)


def sample_data(args):
    image_size = 64  # Ensure this matches your tile resolution
    full_data = TileDataset(data_dir=f'{BASE_DIR}/data/tiles', image_size=image_size)

    train_loader: DataLoader = DataLoader(
        full_data,
        batch_size=args.n_samples,
        shuffle=True,
        num_workers=1,
    )

    print(f'Assuming mean 0 and std 1!')
    x = next(iter(train_loader))
    save_images_split(
        (x + 1) / 2,
        labels=['unlabeled'] * args.n_samples,
        root=f'{BASE_DIR}/training/examples'
    )


if __name__ == '__main__':
    main()
