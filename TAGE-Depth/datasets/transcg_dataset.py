from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from utils.io import read_rgb, read_depth, read_mask
from utils.priors import build_boundary_prior, build_raw_reliability


class TransCGDataset(Dataset):
    """Generic TransCG-style dataset loader.

    This loader expects sample ids in split files and searches for modality files
    under standard subdirectories. Modify `_find_file` if your dataset uses a
    different naming convention.
    """

    def __init__(self, root: str, split_file: str, image_size: Tuple[int, int] = (240, 320), depth_scale: float = 1.0):
        self.root = Path(root)
        self.image_size = tuple(image_size)
        self.depth_scale = float(depth_scale)
        split_path = self.root / split_file
        if not split_path.exists():
            raise FileNotFoundError(f'Split file not found: {split_path}')
        self.ids = [line.strip() for line in split_path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def _find_file(self, subdir: str, sample_id: str):
        candidates = []
        for suffix in ['.npy', '.png', '.tif', '.tiff', '.jpg', '.jpeg']:
            candidates.append(self.root / subdir / f'{sample_id}{suffix}')
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f'Cannot find {subdir}/{sample_id} with supported suffixes')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.ids[idx]
        rgb = read_rgb(self._find_file('rgb', sid), self.image_size)
        raw = read_depth(self._find_file('raw_depth', sid), self.image_size, self.depth_scale)
        mask = read_mask(self._find_file('mask', sid), self.image_size)
        gt = read_depth(self._find_file('gt_depth', sid), self.image_size, self.depth_scale)
        rel = read_depth(self._find_file('rel_depth', sid), self.image_size, self.depth_scale)
        boundary = build_boundary_prior(mask.unsqueeze(0)).squeeze(0)
        reliability = build_raw_reliability(raw.unsqueeze(0), rel.unsqueeze(0)).squeeze(0)
        valid = (gt > 0).float()
        return {
            'id': sid,
            'rgb': rgb,
            'raw_depth': raw,
            'mask': mask,
            'gt_depth': gt,
            'relative_depth': rel,
            'boundary_prior': boundary,
            'raw_reliability': reliability,
            'valid': valid,
        }
