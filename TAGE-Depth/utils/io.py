from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_rgb(path: str, image_size: Tuple[int, int] = None) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    if image_size is not None:
        h, w = image_size
        img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def read_mask(path: str, image_size: Tuple[int, int] = None) -> torch.Tensor:
    img = Image.open(path).convert('L')
    if image_size is not None:
        h, w = image_size
        img = img.resize((w, h), Image.NEAREST)
    arr = (np.asarray(img).astype(np.float32) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def read_depth(path: str, image_size: Tuple[int, int] = None, depth_scale: float = 1.0) -> torch.Tensor:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix == '.npy':
        arr = np.load(path).astype(np.float32)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(path)
        arr = arr.astype(np.float32) * float(depth_scale)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if image_size is not None:
        h, w = image_size
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(arr).unsqueeze(0)


def save_depth_npy(path: str, depth: torch.Tensor):
    path = Path(path)
    ensure_dir(path.parent)
    arr = depth.detach().cpu().squeeze().numpy().astype(np.float32)
    np.save(path, arr)
