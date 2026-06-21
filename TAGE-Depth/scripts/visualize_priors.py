"""Visualize TAGE-Depth prior maps for a single sample."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.io import read_rgb, read_depth, read_mask
from utils.priors import build_boundary_prior, build_raw_reliability


def show(ax, x, title, cmap=None):
    if torch.is_tensor(x):
        x = x.detach().cpu().squeeze().numpy()
    ax.imshow(x, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--raw-depth', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--relative-depth', required=True)
    parser.add_argument('--output', default='outputs/prior_visualization.png')
    args = parser.parse_args()

    rgb = read_rgb(args.rgb).permute(1, 2, 0)
    raw = read_depth(args.raw_depth)
    mask = read_mask(args.mask)
    rel = read_depth(args.relative_depth)
    bnd = build_boundary_prior(mask.unsqueeze(0)).squeeze(0)
    conf = build_raw_reliability(raw.unsqueeze(0), rel.unsqueeze(0)).squeeze(0)
    diff = torch.abs(raw - rel)

    fig, axes = plt.subplots(1, 7, figsize=(18, 3))
    show(axes[0], rgb, 'RGB')
    show(axes[1], raw, 'Raw Depth', 'turbo')
    show(axes[2], mask, 'Mask', 'gray')
    show(axes[3], bnd, 'Boundary Prior', 'gray')
    show(axes[4], rel, 'Relative Depth', 'turbo')
    show(axes[5], conf, 'Raw Reliability', 'magma')
    show(axes[6], diff, 'Relative-Raw Difference', 'Reds')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
