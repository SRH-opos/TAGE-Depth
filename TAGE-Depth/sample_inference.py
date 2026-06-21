"""Minimal random-input inference example for checking installation."""

import torch

from models import TAGEDepth
from utils.priors import build_boundary_prior, build_raw_reliability


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TAGEDepth(base_channels=16).to(device).eval()
    b, h, w = 1, 240, 320
    rgb = torch.rand(b, 3, h, w, device=device)
    raw = torch.rand(b, 1, h, w, device=device) * 2.0
    mask = (torch.rand(b, 1, h, w, device=device) > 0.7).float()
    rel = torch.rand(b, 1, h, w, device=device) * 2.0
    bnd = build_boundary_prior(mask)
    conf = build_raw_reliability(raw, rel)
    with torch.no_grad():
        out = model(rgb, raw, mask, rel, bnd, conf)
    print('pred_depth:', tuple(out['pred_depth'].shape))
    print('final_depth:', tuple(out['final_depth'].shape))


if __name__ == '__main__':
    main()
