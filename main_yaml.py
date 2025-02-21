import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import utils, transforms
from tqdm import tqdm

import datasets
from configs import BaseConfigs, DiffusionConfigs
from trainer import DiffusionTrainer
from misc import reset_seed, write_label_under_batch, save_images_split, parse_paths, find_corresponding_file, \
    load_img_path_to_tensor, broadcast_to_batch
from unetG import UNetModel


def main():
    parser = argparse.ArgumentParser(description='Train a (conditional) diffusion model and sample from it')
    parser.add_argument('path', type=str, help='Path to the yaml setup file')

    args = parser.parse_args()
    with open(args.path, 'r') as f:
        setup = yaml.safe_load(f)
    # print(setup)
    # exit()

    if setup.get('command') == 'diffusion':
        train_diffusion(setup)
    elif setup.get('command') == 'sampleuc':
        sample_unconditional(setup)

def create_configs(setup: dict):
    # Create new configs object
    configs_type = setup.get('configs', {}).get('type')
    if configs_type == 'DiffusionConfigs':
        configs = DiffusionConfigs(**setup.get('configs', {}).get('params', {}))
    elif configs_type == 'ClassifierConfigs':
        configs = ClassifierConfigs(**setup.get('configs', {}).get('params', {}))
    elif configs_type == 'ClassConditionalConfigs':
        configs = ClassConditionalConfigs(**setup.get('configs', {}).get('params', {}))
    else:
        raise ValueError(f'Unknown configs type: {setup.get("configs", {}).get("type")}')
    return configs


def load_configs(setup: dict, limit_types: list = None):
    # Load configs from checkpoint
    configs_type = setup.get('configs', {}).get('type')
    if configs_type == 'DiffusionConfigs' and (not limit_types or 'DiffusionConfigs' in limit_types):
        configs = DiffusionConfigs.from_checkpoint(**setup.get('configs', {}).get('params', {}))
    else:
        raise ValueError(f'Unknown or invalid configs type: {setup.get("configs", {}).get("type")}')
    return configs


def load_trainer(setup: dict, configs: BaseConfigs, limit_types: list = None, default_type: str = None):
    # Load configs from checkpoint
    trainer_type = setup.get('trainer', {}).get('type')
    trainer_type = default_type if trainer_type is None else trainer_type
    if trainer_type == 'DiffusionTrainer' and (not limit_types or 'DiffusionTrainer' in limit_types):
        assert isinstance(configs, DiffusionConfigs)
        trainer = DiffusionTrainer(configs)
    elif trainer_type == 'PoolGuidedTrainer' and (not limit_types or 'PoolGuidedTrainer' in limit_types):
        assert isinstance(configs, DiffusionConfigs)
        trainer = PoolGuidedTrainer(configs)
    else:
        raise ValueError(f'Unknown or invalid trainer type: {setup.get("trainer", {}).get("type")}')
    return trainer


def train_diffusion(setup: dict):
    # Create new configs object
    configs = create_configs(setup)

    # Save setup file to configs directory
    configs.backup_setup(setup, with_date=True, overwrite=setup.get('overwrite', False))

    if setup.get('phase') == 'train':
        # Load existing model (use model at 'resume_path')
        configs.load_checkpoint()

        # Load base model weights (project name or checkpoint path) to start training from
        if setup.get('configs', {}).get('start_from', None) is not None and configs.start_epoch == 0:
            configs.load_state_dict_only(**setup.get('configs', {}).get('start_from', {}))

        # Overwrite loaded configs with setup (if overwrite is True)
        if setup.get('overwrite', False):
            for param in setup.get('configs', {}).get('params', {}).items():
                setattr(configs, param[0], param[1])

        # Create new trainer object
        trainer = load_trainer(setup, configs, limit_types=['DiffusionTrainer'], default_type='DiffusionTrainer')

        # Create dataset for training
        dataset_type = setup.get('dataset', {}).get('type')
        if hasattr(datasets, dataset_type):
            train_data = getattr(datasets, dataset_type)(**setup.get('dataset', {}).get('params', {}))
        else:
            raise ValueError(f'Unknown dataset type: {dataset_type}')

        # Start training
        trainer.train(train_data, **setup.get('train', {}))

        # Save configs
        configs.save()

    elif setup.get('phase') == 'sample':
        configs.load_checkpoint(latest=True)
        trainer = DiffusionTrainer(configs)
        trainer.sample('result.png')


def sample_unconditional(setup):
    configs = load_configs(setup, limit_types=['DiffusionConfigs'])
    trainer = load_trainer(setup, configs, limit_types=['DiffusionTrainer'], default_type='DiffusionTrainer')

    output_dir = setup.get('output_dir', 'results')
    results_dir = Path(configs.root_dir, output_dir)
    results_dir.mkdir(exist_ok=setup.get('overwrite', False), parents=True)

    n_samples = setup.get('n_samples', 8)
    batch_size = setup.get('batch_size', 8)
    for i in range(0, n_samples, batch_size):
        x = trainer.sample(
            path=Path(output_dir, f'result_{i:04d}.png').as_posix(),
            n_samples=batch_size,
            **setup.get('sample', {})
        )
        save_images_split((x + 1) / 2, labels=['unknown'] * batch_size, root=results_dir.as_posix(), idx_start=i)


if __name__ == '__main__':
    main()
