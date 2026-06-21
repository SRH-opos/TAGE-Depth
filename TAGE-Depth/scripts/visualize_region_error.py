"""Generate a simple region-wise error visualization."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.io import read_rgb, read_depth, read_mask
from utils.priors import build_boundary_prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--raw-depth', required=True)
    parser.add_argument('--pred-depth', required=True)
    parser.add_argument('--gt-depth', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--output', default='outputs/region_error.png')
    args = parser.parse_args()

    rgb = read_rgb(args.rgb).permute(1, 2, 0)
    raw = read_depth(args.raw_depth)
    pred = read_depth(args.pred_depth)
    gt = read_depth(args.gt_depth)
    mask = read_mask(args.mask)
    bnd = build_boundary_prior(mask.unsqueeze(0)).squeeze(0)
    raw_err = torch.abs(raw - gt) * mask
    pred_err = torch.abs(pred - gt) * mask
    improve = (raw_err - pred_err).clamp_min(0.0)

    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    items = [
        (rgb, 'RGB', None),
        (mask, 'Mask', 'gray'),
        (bnd, 'Boundary', 'gray'),
        (raw_err, 'Raw Error', 'magma'),
        (pred_err, 'TAGE Error', 'magma'),
        (improve, 'Improvement', 'Blues'),
    ]
    for ax, (img, title, cmap) in zip(axes, items):
        if torch.is_tensor(img):
            img = img.detach().cpu().squeeze().numpy()
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
