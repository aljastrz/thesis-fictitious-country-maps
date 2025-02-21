import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set

import torch
import yaml
from torch import nn

from models import DenoisingDiffusion
from unetG import UNetModel

# This implementation is based on open source resources, provided by Leon Pielage


class BaseConfigs:

    def __init__(self, name: Optional[str] = None, root: str = 'projects', config_path: str = 'config.json'):
        self._root = root
        self.name: Optional[str] = name
        self._config_path: str = config_path
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> str:
        if self.name is None:
            return self._root
        else:
            return Path(self._root, self.name).as_posix()

    @property
    def config_path(self) -> str:
        return Path(self.root_dir, self._config_path).as_posix()

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config

    @classmethod
    def from_json_file(cls, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self):
        config_dict = {}
        for key, value in self.__dict__.items():
            if type(value) in [int, float, str, bool, list, tuple, dict]:
                config_dict[key] = value
        return config_dict

    def to_string(self, short: bool = False):
        config_str = ''
        for key, value in self.__dict__.items():
            if type(value) in [int, float, str, bool, list, tuple, dict]:
                config_str += f'{key} = {value}\n'
            elif not short:
                config_str += f'{key} = {type(value)}\n'
            else:
                config_str += f'{key} = {type(value).__name__}\n'
        return config_str

    def to_json_file(self, json_path: str, dict_to_save: Optional[dict] = None):
        config_dict = self.to_dict() if dict_to_save is None else dict_to_save
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)

    def save(self):
        self.to_json_file(self.config_path)

    def load(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self, key, value)

    def backup_setup(self, setup: dict, with_date: bool = True, overwrite: bool = False):
        setup_name = 'setup'
        if with_date:
            setup_name += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        setup_name += '.yaml'
        setup_path = Path(self.root_dir, setup_name)
        if not setup_path.exists() or overwrite:
            with open(setup_path, 'w') as f:
                yaml.dump(setup, f)


class ModelConfigs(BaseConfigs):

    def __init__(self,
                 *args,
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 epochs: int = 10,
                 seed: int = None,

                 lr: float = 2e-5,
                 weight_decay: float = 1e-5,
                 lr_scheduler_factor: float = 0.5,
                 lr_scheduler_patience: int = 50,

                 log_interval: int = 100,       # how many batches to wait before logging training status
                 val_interval: int = 1,         # how many epochs to wait before evaluating on the validation set
                 sample_interval: int = 1,      # how many epochs to wait before generating samples
                 save_interval: int = 1,        # how many epochs to wait before saving a checkpoint
                 save_latest: bool = True,      # whether to save the latest checkpoint
                 save_dir: str = 'checkpoints',
                 resume_file_name: str = 'checkpoint.pth',
                 save_file_name: str = 'checkpoint.pth',
                 start_epoch: int = 0,
                 n_workers: int = 4,
                 log_dir: str = 'logs',
                 log_file_name: str = 'log.txt',
                 sample_dir: str = 'samples',
                 sample_file_name: str = 'sample.png',

                 model: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if device is None:
            self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device: str = device
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.seed: int = seed

        # Optimizer parameters
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.lr_scheduler_factor: float = lr_scheduler_factor
        self.lr_scheduler_patience: int = lr_scheduler_patience

        # Training parameters
        self.log_interval: int = log_interval
        self.val_interval: int = val_interval
        self.sample_interval: int = sample_interval
        self.save_interval: int = save_interval
        self.save_latest: bool = save_latest
        self._save_dir: str = save_dir
        self._resume_file_name: str = resume_file_name
        self._save_file_name: str = save_file_name
        self.start_epoch: int = start_epoch
        self.n_workers: int = n_workers
        self._log_dir: str = log_dir
        self._log_file_name: str = log_file_name
        self._sample_dir: str = sample_dir
        self._sample_file_name: str = sample_file_name

        self.init_objects(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

        self.prepare_dirs()

    @property
    def save_dir(self):
        return Path(self.root_dir, self._save_dir).as_posix()

    @property
    def resume_path(self):
        return Path(self.save_dir, self._resume_file_name).as_posix()

    @property
    def save_file(self):
        return Path(self.save_dir, self._save_file_name).as_posix()

    @property
    def log_dir(self):
        return Path(self.root_dir, self._log_dir).as_posix()

    @property
    def log_file(self):
        return Path(self.log_dir, self._log_file_name).as_posix()

    @property
    def sample_dir(self):
        return Path(self.root_dir, self._sample_dir).as_posix()

    @property
    def sample_file(self):
        return Path(self.sample_dir, self._sample_file_name).as_posix()

    def prepare_dirs(self):
        if not Path(self.save_dir).exists():
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        if not Path(self.log_dir).exists():
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if not Path(self.sample_dir).exists():
            Path(self.sample_dir).mkdir(parents=True, exist_ok=True)

    def init_objects(self, model: nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler):
        # Create models, optimizer and learning rate scheduler
        self.model: nn.Module = model
        if self.model is not None:
            self.model.to(self.device)

        if optimizer is None and self.model is not None:
            self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer: torch.optim.Optimizer = optimizer

        if lr_scheduler is None and self.optimizer is not None:
            self.lr_scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
        else:
            self.lr_scheduler: torch.optim.lr_scheduler = lr_scheduler

    @staticmethod
    def postfix_filepath(filepath: str, postfix: str) -> str:
        path = Path(filepath).parent
        stem = Path(filepath).stem
        ext = Path(filepath).suffix
        filename = f'{stem}_{postfix}{ext}'
        filepath = Path(path, filename)
        return filepath.as_posix()

    def timestamp_specific_filepath(self, filepath: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.postfix_filepath(filepath, timestamp)

    def epoch_specific_filepath(self, filepath: str, epoch: int = None) -> str:
        path = Path(filepath).parent
        stem = Path(filepath).stem
        ext = Path(filepath).suffix
        if epoch is None:
            epoch = self.start_epoch - 1
        filename = f'{stem}-{epoch:04d}{ext}'
        filepath = Path(path, filename)
        return filepath.as_posix()

    def latest_checkpoint_path(self) -> Optional[str]:
        latest_path = self._latest_checkpoint_path(self.save_dir)
        # print(f'[DEBUG] Latest checkpoint path resolved: {latest_path}')
        return latest_path

    @staticmethod
    def _latest_checkpoint_path(save_dir: str) -> Optional[str]:
        # print(f'[DEBUG] Checking for checkpoints in directory: {save_dir}')
        checkpoints = list(filter(lambda x: '-' in x.as_posix(), Path(save_dir).glob('*.pth')))
        # print(f'[DEBUG] Found checkpoints: {checkpoints}')
        if checkpoints is not None and len(checkpoints) > 0:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('-')[-1])).as_posix()
            # print(f'[DEBUG] Latest checkpoint selected: {latest_checkpoint}')
            return latest_checkpoint
        # print('[DEBUG] No valid checkpoints found in the directory.')
        return None

    def load_checkpoint(self, path: str = None, latest: bool = False):
        if path is None:
            path = self.resume_path
        if latest:
            path = self.latest_checkpoint_path()
        # print(f'[DEBUG] Attempting to load checkpoint from path: {path}')
        if path is None or not Path(path).is_file():
            # print(f'[DEBUG] No checkpoint found at the specified path: {path}')
            return
        try:
            checkpoint = torch.load(path)
            self.load_checkpoint_dict(checkpoint)
            # print(f'[DEBUG] Successfully loaded checkpoint from: {path}')
        except Exception as e:
            print(f'[DEBUG] Failed to load checkpoint. Error: {e}')

    def save_checkpoint(self, path: str = None, epoch_specific: bool = False):
        if path is None:
            path = self.save_file
        if epoch_specific:
            path = self.epoch_specific_filepath(path)
        # print(f'[DEBUG] Saving checkpoint to: {path}')
        checkpoint = self.get_checkpoint_dict()
        try:
            torch.save(checkpoint, path)
            # print(f'[DEBUG] Successfully saved checkpoint at: {path}')
        except Exception as e:
            print(f'[DEBUG] Failed to save checkpoint. Error: {e}')

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str = None,
                        name: Optional[str] = None, root: str = 'projects', save_dir: str = 'checkpoints'):
        checkpoint_path = cls._get_checkpoint_path(checkpoint_path, name, root, save_dir)
        # print(f'[DEBUG] Loading checkpoint from path: {checkpoint_path}')
        try:
            checkpoint = torch.load(checkpoint_path)
            configs_dict = {}
            for key, value in checkpoint['configs'].items():
                if key == 'root_dir':
                    configs_dict['root'] = Path(value).parent.as_posix()
                elif key == 'config_path':
                    configs_dict['config_path'] = Path(value).name
                elif key == 'save_dir':
                    configs_dict['save_dir'] = Path(value).name
                elif key == 'resume_path':
                    configs_dict['resume_file_name'] = Path(value).name
                elif key == 'save_file':
                    configs_dict['save_file_name'] = Path(value).name
                elif key == 'log_dir':
                    configs_dict['log_dir'] = Path(value).name
                elif key == 'log_file':
                    configs_dict['log_file_name'] = Path(value).name
                elif key == 'sample_dir':
                    configs_dict['sample_dir'] = Path(value).name
                elif key == 'sample_file':
                    configs_dict['sample_file_name'] = Path(value).name
                elif key.startswith('_'):
                    configs_dict[key[1:]] = value
                else:
                    configs_dict[key] = value
            configs = cls(**configs_dict)
            configs.model.load_state_dict(checkpoint['model_state_dict'])
            configs.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            configs.lr_scheduler.load_state_dict(checkpoint['learning_rate_scheduler_state_dict'])
            # print(f'[DEBUG] Successfully loaded checkpoint configurations from: {checkpoint_path}')
            return configs
        except Exception as e:
            # print(f'[DEBUG] Failed to load checkpoint from: {checkpoint_path}. Error: {e}')
            return None

    def get_checkpoint_dict(self):
        checkpoint = {
            'configs': self.to_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        # print(f'[DEBUG] Generated checkpoint dictionary.')
        return checkpoint

    def load_checkpoint_dict(self, checkpoint: dict):
        # print(f'[DEBUG] Loading checkpoint dictionary into current configuration.')
        try:
            self.__dict__.update(checkpoint['configs'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['learning_rate_scheduler_state_dict'])
            # print(f'[DEBUG] Successfully loaded all components from checkpoint dictionary.')
        except Exception as e:
            print(f'[DEBUG] Failed to load checkpoint dictionary. Error: {e}')

    @staticmethod
    def _get_checkpoint_path(checkpoint_path: str = None,
                            name: Optional[str] = None, root: str = 'projects', save_dir: str = 'checkpoints'):
        # print(f'[DEBUG] Resolving checkpoint path with checkpoint_path: {checkpoint_path}, name: {name}, root: {root}, save_dir: {save_dir}')
        try:
            if checkpoint_path is None:
                save_dir = Path(root, name, save_dir).as_posix()
                checkpoint_path = ModelConfigs._latest_checkpoint_path(save_dir)
            if checkpoint_path is None:
                checkpoint_path = Path(save_dir, 'checkpoint.pth').as_posix()
            # print(f'[DEBUG] Resolved checkpoint path: {checkpoint_path}')
            return checkpoint_path
        except Exception as e:
            # print(f'[DEBUG] Error resolving checkpoint path. Error: {e}')
            return None


    @property
    def current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @current_lr.setter
    def current_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class DiffusionConfigs(ModelConfigs):

    def __init__(self,
                 *args,
                 n_steps: int = 1_000,
                 n_samples: int = 16,
                 image_size: int = 64,
                 image_channels: int = 3,
                 dim: int = 64,
                 dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 diffusion: DenoisingDiffusion = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps: int = n_steps
        self.n_samples: int = n_samples
        self.image_size: int = image_size
        self.image_channels: int = image_channels
        self.dim: int = dim
        self.dim_mults: Tuple[int, ...] = dim_mults

        # Create models, optimizer and learning rate scheduler
        if self.model is None:
            self.model: nn.Module = UNetModel(
                image_size=self.image_size,
                in_channels=self.image_channels,
                model_channels=self.dim,
                out_channels=self.image_channels,
                attention_resolutions=(32, 16, 8),
                channel_mult=self.dim_mults,
                num_res_blocks=2,
            )

        if diffusion is None:
            self.diffusion: DenoisingDiffusion = DenoisingDiffusion(
                eps_model=self.model,
                n_steps=self.n_steps,
                device=self.device,
            )
        else:
            self.diffusion: DenoisingDiffusion = diffusion

        self.init_objects(model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
