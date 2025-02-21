import logging
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import utils
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from configs import ModelConfigs, DiffusionConfigs
from misc import write_label_under_batch, reset_seed, save_images_split

# This implementation is based on open source resources, provided by Leon Pielage


class BaseTrainer:
    """
    Abstract base class for training models.
    Step functions for training, validation and sampling must be implemented in child classes.
    """

    def __init__(self, configs: ModelConfigs) -> None:
        """
        Initialize the base trainer with the given model configurations and set up logging

        :param configs: The model configurations
        """
        self.configs: ModelConfigs = configs
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(self.configs.timestamp_specific_filepath(self.configs.log_file)),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger('Trainer' if self.configs.name is None else self.configs.name)
        self.logger.info(f'Configs:\n{self.configs.to_string(short=True)}')

    def _train_validate_split(self, dataset: Dataset, val_size: float = 0.1, **kwargs) -> tuple[Dataset, Dataset]:

        """
        Split a dataset into training and validation sets

        :param dataset: The dataset to split
        :param val_size: The size of the validation set as a fraction of the whole dataset
        :param kwargs: Additional arguments for child class implementations
        :return: The training and validation datasets
        """
        all_size = len(dataset)
        val_size = int(all_size * val_size)
        train_size = all_size - val_size
        rs_kwargs = {
            'generator': torch.Generator().manual_seed(self.configs.seed)
        } if self.configs.seed is not None else {}
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], **rs_kwargs)
        return train_dataset, val_dataset

    def _train_init_dataloader(self, dataset: Dataset, val_dataset: Optional[Dataset] = None, **kwargs) \
        -> tuple[DataLoader, Optional[DataLoader]]:

        """
        Initialize the training and validation dataloaders

        :param dataset: The training dataset
        :param val_dataset: The validation dataset (optional)
        :param kwargs: Additional arguments for child class implementations
        :return: The training and validation dataloaders
        """
        train_loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.configs.batch_size,
            shuffle=True,
            num_workers=self.configs.n_workers,
        )
        val_loader: Optional[DataLoader] = None
        if val_dataset is not None:
            val_loader: DataLoader = DataLoader(
                val_dataset,
                batch_size=self.configs.batch_size,
                shuffle=False,
                num_workers=self.configs.n_workers,
            )
            self.logger.info(f'Validation on {type(val_loader.dataset).__name__}\n{str(val_loader.dataset)}\n')
        return train_loader, val_loader

    def _train_init_hook(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs) -> None:
        """
        Additional initialization hook for child class implementations
        """
        pass

    def _train_epoch_train(self, train_loader: DataLoader, epoch: int, class_weight: torch.Tensor = None,
                           **kwargs) -> None:
        """
        Train the model for one epoch by iterating over the training dataloader.
        The training per batch is done by the train_step function of the child class implementation.

        :param train_loader: The training dataloader
        :param epoch: The current epoch number
        :param class_weight: The class weights for the loss function (optional)
        :param kwargs: Additional arguments for child class implementations
        """
        self.configs.model.train()
        for batch_idx, (data, label) in enumerate(pbar := tqdm(train_loader, desc='Training')):
            data, label = data.to(self.configs.device), label.to(self.configs.device)

            loss = self.train_batch(data, label, class_weight=class_weight, **kwargs)

            pbar.set_description(f'Training | Epoch: {epoch:04d} | Loss: {loss.item():.4f}')
            if batch_idx % self.configs.log_interval == 0:
                self.logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                                 f'({100. * batch_idx / len(train_loader):3.0f}%)] Loss: {loss.item():.6f}')

    def _train_epoch_validate(self, val_loader: DataLoader, epoch: int, **kwargs) -> None:
        """
        Validate the model at intervals during training and adjust the learning rate if a scheduler is used

        :param val_loader: The validation dataloader
        :param epoch: The current epoch number
        :param kwargs: Additional arguments for child class implementations
        """
        if self.configs.val_interval != 0 and epoch % self.configs.val_interval == 0:
            val_loss, _ = self.validate(val_loader)
            if self.configs.lr_scheduler is not None:
                self.configs.lr_scheduler.step(val_loss)
            self.logger.info(f'Learning rate: {self.configs.current_lr:.8f}')

    def _train_epoch_save(self, epoch: int, **kwargs) -> None:
        """
        Save the model and samples at intervals during training

        :param epoch: The current epoch number
        :param kwargs: Additional arguments for child class implementations
        """
        self.configs.start_epoch = epoch + 1
        if self.configs.save_interval != 0 and epoch % self.configs.save_interval == 0:
            self.configs.save_checkpoint(epoch_specific=True)
        if self.configs.save_latest:
            self.configs.save_checkpoint(epoch_specific=False)
        if self.configs.sample_interval != 0 and epoch % self.configs.sample_interval == 0:
            self.sample()

    def train_batch(self, data: torch.Tensor, label: torch.Tensor, class_weight: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        """
        Training step function for child class implementations
        """
        raise NotImplementedError

    def train(self, train_dataset: Dataset, val_dataset: Dataset = None, auto_val_size: float = None, **kwargs) -> None:
        """
        Train the model on the given dataset

        :param train_dataset: The training dataset
        :param val_dataset: The validation dataset (optional)
        :param auto_val_size: The size of the validation set as a fraction of the whole dataset (optional)
        :param kwargs: Additional arguments for child class implementations
        """
        self.logger.info(f'Training on {type(train_dataset).__name__}\n{str(train_dataset)}\n')

        if auto_val_size is not None:
            train_dataset, val_dataset = self._train_validate_split(train_dataset, val_size=auto_val_size, **kwargs)

        if hasattr(train_dataset, 'class_weight'):
            class_weight = train_dataset.class_weight.to(self.configs.device)
        else:
            class_weight = None

        train_loader, val_loader = self._train_init_dataloader(train_dataset, val_dataset, **kwargs)

        self._train_init_hook(train_loader, val_loader, **kwargs)

        self.configs.model.to(self.configs.device)

        with logging_redirect_tqdm():
            for epoch in range(self.configs.start_epoch, self.configs.epochs):
                self._train_epoch_train(train_loader, epoch=epoch, class_weight=class_weight, **kwargs)
                if val_loader is not None:
                    self._train_epoch_validate(val_loader=val_loader, epoch=epoch, **kwargs)
                self._train_epoch_save(epoch, **kwargs)

    def validate_batch(self, data: torch.Tensor, label: torch.Tensor, batch_idx: Optional[int] = None,
                   **kwargs) -> tuple[torch.Tensor, dict]:

        """
        Validation step function for child class implementations
        """
        raise NotImplementedError

    def validate(self, val_loader: DataLoader) -> tuple[float, dict]:
        """
        Validate the model on the given dataloader by iterating over the batches and calculating the loss and
        other metrics.

        :param val_loader: The validation dataloader
        :return: The average validation loss and the cumulative metrics as a dictionary
        """
        self.configs.model.eval()
        val_loss = 0
        cum_metrics = dict()
        with torch.no_grad():
            with logging_redirect_tqdm():
                for batch_idx, (data, label) in enumerate(pbar := tqdm(val_loader, desc='Validation')):
                    data, label = data.to(self.configs.device), label.to(self.configs.device)
                    loss, metric_dict = self.validate_batch(data, label, batch_idx)
                    val_loss += loss.item()
                    for k, v in metric_dict.get('cum_metrics', {}).items():
                        if k not in cum_metrics:
                            cum_metrics[k] = 0
                        cum_metrics[k] += v.item()

                    pbar.set_description(f'Validation | Loss: {loss.item():.4f}')
                    if batch_idx % self.configs.log_interval == 0:
                        self.logger.info(f'Validation: [{batch_idx * len(data)}/{len(val_loader.dataset)} '
                                         f'({100. * batch_idx / len(val_loader):3.0f}%)] Loss: {loss.item():.6f}')
        cum_metrics = {k: v / len(val_loader) for k, v in cum_metrics.items()}
        cum_metric_str = ', '.join(f'{k}: {v:.4f}' for k, v in cum_metrics.items())
        self.logger.info(f'Validation: {cum_metric_str}')
        self.configs.model.train()
        return val_loss / len(val_loader), cum_metrics

    def _sample_save_path(self, path: str = None, postfix: str = None) -> str:
        """
        Get the save path for samples

        :param path: The path to save the samples to relative to project root (optional)
        :param postfix: The postfix to add to the sample file name (optional)
        """
        if path is None:
            path = self.configs.epoch_specific_filepath(self.configs.sample_file)
            if postfix is not None:
                path = self.configs.postfix_filepath(path, postfix)
        else:
            path = Path(self.configs.root_dir, path).as_posix()
        return path

    def sample_batch(self, path: str, n_samples: int = None, **kwargs) -> torch.Tensor:
        """
        Sample step function for child class implementations
        """
        raise NotImplementedError

    def sample(self, path: str = None, n_samples: int = None, **kwargs):
        """
        Sample from the model and save the samples to the given path or the default sample file path

        :param path: The path to save the samples to relative to project root (optional)
        :param n_samples: The number of samples to generate (optional)
        """
        path = self._sample_save_path(path, kwargs.get('postfix', None))

        # Sample from the model
        with logging_redirect_tqdm(), torch.no_grad():
            self.configs.model.eval()
            x = self.sample_batch(path, n_samples=n_samples, **kwargs)
        return x


class DiffusionTrainer(BaseTrainer):
    """
    Trainer for Denoising Diffusion Probabilistic Models
    """

    def __init__(self, configs: DiffusionConfigs) -> None:
        """
        Initialize the diffusion trainer with the given diffusion configurations

        :param configs: The diffusion configurations
        """
        super().__init__(configs)
        self.configs: DiffusionConfigs = configs

    def train_batch(self, data: torch.Tensor, label: torch.Tensor, class_weight: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        """
        Train the model by optimizing the diffusion loss

        :param data: The input data batch
        :param label: The label batch (not used)
        :param class_weight: The class weights for the loss function (not used)
        :param kwargs: Additional arguments (not used)
        :return: The diffusion loss
        """
        self.configs.optimizer.zero_grad()
        loss = self.configs.diffusion.loss(data)
        loss.backward()
        self.configs.optimizer.step()
        return loss

    def validate_batch(self, data: torch.Tensor, label: torch.Tensor, batch_idx: int = None,
                   **kwargs) -> tuple[torch.Tensor, dict]:

        """
        Validate the model by calculating the diffusion loss

        :param data: The input data batch
        :param label: The label batch (not used)
        :param batch_idx: The batch index (not used)
        :param kwargs: Additional arguments (not used)
        :return: The diffusion loss and an empty dictionary as no further metrics are calculated
        """
        return self.configs.diffusion.loss(data), {}

    def save_samples(self, x: torch.Tensor, path: str, postfix: str = '', labels: torch.Tensor = None) -> None:
        """
        Save the given samples. Add labels under the samples if given.

        :param x: The samples to save as a tensor in the range [-1, 1]
        :param path: The path to save the samples to relative to project root
        :param postfix: The postfix to add to the sample file name (optional)
        :param labels: The labels to add under the samples (optional)
        """
        path = self._sample_save_path(path, postfix)

        x_norm = (x + 1) / 2
        if labels is not None:
            x_norm = write_label_under_batch(x_norm, labels)
        utils.save_image(x_norm, path)

    def _sample_batch_setup(self, **kwargs) -> dict:
        """
        Additional setup for the sample_batch function for child class implementations
        """
        return dict()

    def _sample_batch_update_x(self, x: torch.Tensor, t: torch.Tensor, setup_out: dict, *, n_samples: int,
                               **kwargs) -> torch.Tensor:
        """
        Update the samples for the given time step t for the sample_batch function
        """
        x = self.configs.diffusion.p_sample(
            x,
            torch.full((n_samples,), t.item(), dtype=torch.long, device=self.configs.device),
        )
        return x



    def sample_batch(self,
                 path: str = None,
                 n_samples: int = None, *,
                 latent_batch: torch.Tensor = None,
                 latent_without_noise: bool = False,
                 left_previous: torch.Tensor = None,
                 top_previous: torch.Tensor = None,
                 start_step: int = 0,
                 save_intermediate_step: int = None,
                 save_result: bool = True,
                 output_dict: dict = None,
                 output_dict_all_steps: bool = False,
                 seed: int = None,
                 **kwargs) -> torch.Tensor:
        """
        Sample from the model, ensuring better coherence between tiles by improving overlap handling.

        :param left_previous: A tensor representing the left overlap from the previous tile.
        :param top_previous: A tensor representing the top overlap from the tile above.
        """
        assert start_step >= 0, "Start step must be >= 0"
        assert not save_result or (save_result and path is not None), "Path must be given if save_result is True"

        # Set number of samples
        n_samples = n_samples or self.configs.n_samples

        # Initialize noise or use given latent batch
        xT = self.configs.diffusion.normal(
            (n_samples, self.configs.image_channels, self.configs.image_size, self.configs.image_size),
            seed=seed
        )
        
        if latent_batch is None:
            x = xT.clone()
        else:
            x = latent_batch.to(self.configs.device)
            if not latent_without_noise:
                t = self.configs.n_steps - start_step - 1
                t = torch.tensor([t], dtype=torch.long, device=self.configs.device)
                x = self.configs.diffusion.q_sample(x, t=t, eps=xT)

        reset_seed(seed)

        # Ensure output_dict tracks progress
        if output_dict is not None:
            output_dict['progress'] = 0.0

        setup_out = self._sample_batch_setup(
            path=path, n_samples=n_samples, latent_batch=latent_batch, latent_without_noise=latent_without_noise,
            start_step=start_step, save_intermediate_step=save_intermediate_step, save_result=save_result,
            output_dict=output_dict, output_dict_all_steps=output_dict_all_steps, seed=seed, **kwargs
        )

        for i in tqdm(range(start_step, self.configs.n_steps), desc='Sampling'):
            t = self.configs.n_steps - i - 1
            t = torch.tensor([t], dtype=torch.long, device=self.configs.device)

            # **Improve Left Overlap Handling**
            if left_previous is not None:
                noisy_left = self.configs.diffusion.q_sample(
                    left_previous, t=t, eps=xT[:, :, :, :left_previous.size(-1)]
                )
                blend_width = left_previous.size(-1)

                # 🔥 Instead of averaging, we *conditionally guide* the left region
                alpha = torch.linspace(0, 1, blend_width, device=self.configs.device).view(1, 1, 1, blend_width)
                x[:, :, :, :blend_width] = x[:, :, :, :blend_width] * (1 - alpha) + noisy_left * alpha

            # **Improve Top Overlap Handling**
            if top_previous is not None:
                noisy_top = self.configs.diffusion.q_sample(
                    top_previous, t=t, eps=xT[:, :, :top_previous.size(-2), :]
                )
                blend_height = top_previous.size(-2)

                # 🔥 Instead of averaging, *conditionally guide* the top region
                alpha = torch.linspace(0, 1, blend_height, device=self.configs.device).view(1, 1, blend_height, 1)
                x[:, :, :blend_height, :] = x[:, :, :blend_height, :] * (1 - alpha) + noisy_top * alpha

            # **Diffusion Step**
            x = self._sample_batch_update_x(
                x, t, setup_out=setup_out, path=path, n_samples=n_samples,
                latent_batch=latent_batch, latent_without_noise=latent_without_noise,
                start_step=start_step, save_intermediate_step=save_intermediate_step,
                save_result=save_result, output_dict=output_dict,
                output_dict_all_steps=output_dict_all_steps, seed=seed, **kwargs
            )

            # **Save intermediate step if needed**
            if save_intermediate_step is not None and i % save_intermediate_step == 0:
                x_norm = (x + 1) / 2
                if output_dict is not None:
                    if output_dict_all_steps:
                        output_dict[f'step_{i:04d}'] = x_norm.cpu().detach().clamp(0, 1)
                    else:
                        output_dict['latest'] = x_norm.cpu().detach().clamp(0, 1)

            # Update progress
            if output_dict is not None:
                output_dict['progress'] = (i + 1) / self.configs.n_steps

        # Final output
        x_norm = (x + 1) / 2
        if save_result:
            save_image(x_norm, path)
        if output_dict is not None:
            output_dict['final'] = x_norm.cpu().detach().clamp(0, 1)

        return x


