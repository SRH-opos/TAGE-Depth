import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from datasets import TransCGDataset
from models import TAGEDepth
from utils.losses import TAGEDepthLoss
from utils.metrics import depth_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tage_depth.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 6248)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_cfg = cfg['data']
    train_set = TransCGDataset(
        data_cfg['root'], data_cfg['train_split'],
        image_size=tuple(data_cfg['image_size']), depth_scale=float(data_cfg.get('depth_scale', 1.0))
    )
    val_set = TransCGDataset(
        data_cfg['root'], data_cfg['val_split'],
        image_size=tuple(data_cfg['image_size']), depth_scale=float(data_cfg.get('depth_scale', 1.0))
    )
    train_loader = DataLoader(train_set, batch_size=cfg['training']['batch_size'], shuffle=True,
                              num_workers=cfg['training'].get('num_workers', 0), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False,
                            num_workers=cfg['training'].get('num_workers', 0), pin_memory=True)

    model = TAGEDepth(
        base_channels=cfg['model'].get('base_channels', 48),
        num_scales=cfg['model'].get('num_scales', 4),
        max_depth=cfg['model'].get('max_depth', 10.0),
    ).to(device)

    loss_fn = TAGEDepthLoss(cfg.get('loss', {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr_adapter']),
                                  weight_decay=float(cfg['training']['weight_decay']))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['training'].get('amp', True)) and device.type == 'cuda')
    out_dir = Path(cfg['output']['out_dir'])
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    for epoch in range(1, int(cfg['training']['epochs']) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(
                    batch['rgb'], batch['raw_depth'], batch['mask'],
                    batch['relative_depth'], batch['boundary_prior'], batch['raw_reliability']
                )
                loss_dict = loss_fn(outputs, batch)
                loss = loss_dict['loss']
            scaler.scale(loss).backward()
            if cfg['training'].get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['training']['grad_clip']))
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_to_device(batch, device)
                outputs = model(
                    batch['rgb'], batch['raw_depth'], batch['mask'],
                    batch['relative_depth'], batch['boundary_prior'], batch['raw_reliability']
                )
                vals.append(depth_metrics(outputs['final_depth'], batch['gt_depth'], batch['mask'], batch['valid']))
        mean_val = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        print(f'[Val] epoch={epoch}: {mean_val}')
        score = mean_val['mae'] + mean_val['rmse'] + mean_val['rel']
        ckpt = {'epoch': epoch, 'model': model.state_dict(), 'config': cfg, 'metrics': mean_val}
        torch.save(ckpt, ckpt_dir / 'last.pth')
        if score < best_val:
            best_val = score
            torch.save(ckpt, ckpt_dir / 'best.pth')


if __name__ == '__main__':
    main()
