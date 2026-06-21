from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import build_boundary_prior


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.float()
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def masked_rmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mean((pred - gt) ** 2, mask) + 1e-12)


def gradient_x(d: torch.Tensor) -> torch.Tensor:
    return d[:, :, :, 1:] - d[:, :, :, :-1]


def gradient_y(d: torch.Tensor) -> torch.Tensor:
    return d[:, :, 1:, :] - d[:, :, :-1, :]


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    lx = masked_mean(torch.abs(gradient_x(pred) - gradient_x(gt)), mx)
    ly = masked_mean(torch.abs(gradient_y(pred) - gradient_y(gt)), my)
    return 0.5 * (lx + ly)


def hard_pixel_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    err = torch.abs(pred - gt)
    vals = []
    for b in range(pred.shape[0]):
        e = err[b][mask[b] > 0.5]
        if e.numel() == 0:
            continue
        k = max(1, int(e.numel() * ratio))
        vals.append(torch.topk(e.reshape(-1), k=k, largest=True).values.mean())
    if not vals:
        return pred.new_tensor(0.0)
    return torch.stack(vals).mean()


class TAGEDepthLoss(nn.Module):
    """Training objective for TAGE-Depth."""

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.w = weights

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = outputs['final_depth']
        gt = batch['gt_depth']
        raw = batch['raw_depth']
        mask = batch['mask'].float().clamp(0.0, 1.0)
        valid = batch.get('valid', torch.ones_like(mask)).float().clamp(0.0, 1.0)
        region = valid * mask
        bg = valid * (1.0 - mask)
        boundary = build_boundary_prior(mask) * valid

        l_mask = masked_mean(torch.abs(pred - gt), region)
        l_rmse = masked_rmse(pred, gt, region)
        l_hard = hard_pixel_loss(pred, gt, region)
        l_bnd = masked_mean(torch.abs(pred - gt), boundary)
        l_grad = gradient_loss(pred, gt, region)
        l_bg = masked_mean(torch.abs(pred - raw), bg)

        # Optional reference-preserving terms.
        ref = batch.get('reference_depth', None)
        if ref is not None:
            ref_err = torch.abs(ref - gt)
            pred_err = torch.abs(pred - gt)
            l_stab = masked_mean(F.relu(pred_err - ref_err), region)
            easy = (ref_err < 0.01).float() * region
            l_pres = masked_mean(torch.abs(pred - ref), easy) if easy.sum() > 0 else pred.new_tensor(0.0)
        else:
            l_stab = pred.new_tensor(0.0)
            l_pres = pred.new_tensor(0.0)

        total = (
            self.w.get('lambda_mask', 1.0) * l_mask
            + self.w.get('lambda_rmse', 1.0) * l_rmse
            + self.w.get('lambda_hard', 0.0) * l_hard
            + self.w.get('lambda_boundary', 0.0) * l_bnd
            + self.w.get('lambda_grad', 0.0) * l_grad
            + self.w.get('lambda_bg', 0.0) * l_bg
            + self.w.get('lambda_stab', 0.0) * l_stab
            + self.w.get('lambda_pres', 0.0) * l_pres
        )
        return {
            'loss': total,
            'mask': l_mask.detach(),
            'rmse': l_rmse.detach(),
            'hard': l_hard.detach(),
            'boundary': l_bnd.detach(),
            'grad': l_grad.detach(),
            'bg': l_bg.detach(),
            'stab': l_stab.detach(),
            'pres': l_pres.detach(),
        }
