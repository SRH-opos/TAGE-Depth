import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from datasets import TransCGDataset
from models import TAGEDepth
from utils.metrics import depth_metrics


def move_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tage_depth.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_file = cfg['data'][f'{args.split}_split']
    dataset = TransCGDataset(cfg['data']['root'], split_file, tuple(cfg['data']['image_size']), cfg['data'].get('depth_scale', 1.0))
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training'].get('num_workers', 0))

    model = TAGEDepth(cfg['model'].get('base_channels', 48), cfg['model'].get('num_scales', 4), cfg['model'].get('max_depth', 10.0)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    model.eval()

    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch = move_to_device(batch, device)
            out = model(batch['rgb'], batch['raw_depth'], batch['mask'], batch['relative_depth'], batch['boundary_prior'], batch['raw_reliability'])
            all_metrics.append(depth_metrics(out['final_depth'], batch['gt_depth'], batch['mask'], batch['valid']))
    mean = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    print(mean)


if __name__ == '__main__':
    main()
