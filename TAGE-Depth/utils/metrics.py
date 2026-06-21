from typing import Dict

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask.float()).sum() / mask.float().sum().clamp_min(eps)


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor = None) -> Dict[str, float]:
    if valid is None:
        valid = torch.ones_like(mask)
    region = (mask.float() > 0.5).float() * (valid.float() > 0.5).float() * (gt > 0).float()
    err = torch.abs(pred - gt)
    mae = masked_mean(err, region)
    rmse = torch.sqrt(masked_mean((pred - gt) ** 2, region) + 1e-12)
    rel = masked_mean(err / gt.clamp_min(1e-6), region)
    ratio = torch.max(pred.clamp_min(1e-6) / gt.clamp_min(1e-6), gt.clamp_min(1e-6) / pred.clamp_min(1e-6))
    d105 = masked_mean((ratio < 1.05).float(), region)
    d110 = masked_mean((ratio < 1.10).float(), region)
    d125 = masked_mean((ratio < 1.25).float(), region)
    return {
        'mae': float(mae.cpu()),
        'rmse': float(rmse.cpu()),
        'rel': float(rel.cpu()),
        'delta_1.05': float(d105.cpu()),
        'delta_1.10': float(d110.cpu()),
        'delta_1.25': float(d125.cpu()),
    }
