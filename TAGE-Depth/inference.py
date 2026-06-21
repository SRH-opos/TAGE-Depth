import argparse
from pathlib import Path

import torch
import yaml

from models import TAGEDepth
from utils.io import read_rgb, read_depth, read_mask, save_depth_npy
from utils.priors import build_boundary_prior, build_raw_reliability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tage_depth.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--raw-depth', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--relative-depth', required=True)
    parser.add_argument('--output', default='outputs/final_depth.npy')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    image_size = tuple(cfg['data']['image_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rgb = read_rgb(args.rgb, image_size).unsqueeze(0).to(device)
    raw = read_depth(args.raw_depth, image_size, cfg['data'].get('depth_scale', 1.0)).unsqueeze(0).to(device)
    mask = read_mask(args.mask, image_size).unsqueeze(0).to(device)
    rel = read_depth(args.relative_depth, image_size, cfg['data'].get('depth_scale', 1.0)).unsqueeze(0).to(device)
    bnd = build_boundary_prior(mask)
    conf = build_raw_reliability(raw, rel)

    model = TAGEDepth(
        base_channels=cfg['model'].get('base_channels', 48),
        num_scales=cfg['model'].get('num_scales', 4),
        max_depth=cfg['model'].get('max_depth', 10.0),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        out = model(rgb, raw, mask, rel, bnd, conf)
    save_depth_npy(args.output, out['final_depth'][0])
    print(f'Saved final depth to: {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
