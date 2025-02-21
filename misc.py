import random
from pathlib import Path
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from PIL import ImageDraw, Image
from torchvision.transforms import transforms

# This implementation is based on open source resources, provided by Leon Pielage

def reset_seed(seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


def write_on_image(img_tensor: torch.Tensor, text: str, pos: Tuple[int, int] = (0, 0)):
    channels, height, width = img_tensor.size()
    image = transforms.ToPILImage()(img_tensor.clip(0, 1))#.convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.text(pos, text, fill=(255,) * channels)
    return transforms.ToTensor()(image)


def write_under_image(img_tensor: torch.Tensor, text: str, text_area_size: int = 16):
    channels, height, width = img_tensor.size()
    img_border = torch.zeros(channels, height + text_area_size, width)
    img_border[:, :height, :] = img_tensor
    image_text = write_on_image(img_border, text, pos=(0, height))
    return image_text


def write_label_under_batch(batch: torch.Tensor, labels: Union[torch.Tensor, list, str], text_area_size: int = 16):
    batch_size, channels, height, width = batch.size()
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    assert batch_size == len(labels), f'Batch size ({batch_size}) and num labels ({len(labels)}) must be equal'
    result = torch.empty(batch_size, channels, height + text_area_size, width)
    for i, img in enumerate(batch):
        result[i] = write_under_image(img, str(labels[i]), text_area_size=text_area_size)
    return result


def save_images_split(images: torch.Tensor, labels: Union[torch.Tensor, list, str], root: str,
                      idx_start: Optional[int] = 0, prefix: str = '', postfix: str = '',
                      prefix_batch: List[str] = None, postfix_batch: List[str] = None):
    batch_size, channels, height, width = images.size()
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    assert batch_size == len(labels), f'Batch size ({batch_size}) and num labels ({len(labels)}) must be equal'
    assert idx_start is not None or prefix_batch is not None or postfix_batch is not None, \
        'idx_start, prefix_batch, and postfix_batch cannot all be None'
    if prefix_batch is None:
        prefix_batch = [''] * batch_size
    if postfix_batch is None:
        postfix_batch = [''] * batch_size
    for i, img in enumerate(images):
        img = transforms.ToPILImage()(img.clip(0, 1))
        label = str(labels[i])
        idx_string = f'{idx_start + i:05d}' if idx_start is not None else ''
        path = Path(root, label, f'{prefix_batch[i]}{prefix}{idx_string}{postfix}{postfix_batch[i]}.png')
        path.parent.mkdir(parents=False, exist_ok=True)
        img.save(path)


def load_img_path_to_tensor(path: Path, height: int = None, width: int = None, resize: bool = False,
                            alpha_value: float = None):
    image = Image.open(path)
    image = image.convert('RGBA')
    image = transforms.ToTensor()(image).float()
    if alpha_value is not None:
        image[3] = alpha_value
    if resize:
        image = transforms.Resize((height, width))(image)
    return image


def broadcast_to_batch(tensor: torch.Tensor, batch_size: int):
    return tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)


def parse_paths(paths: List[Union[str, Path]]) -> List[Path]:
    results = []
    for path in paths:
        path = Path(path)
        results.extend(path.parent.glob(path.name))
    return results


def find_corresponding_file(file: Path, paths: List[Path], none_ok: bool = False) -> Optional[Path]:
    for path in paths:
        if file.stem in path.stem:
            return path
    if none_ok:
        return None
    raise FileNotFoundError(f'No corresponding file found for {file}')
