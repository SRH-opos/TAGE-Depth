from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import EncoderBranch, ConvBlock, ScaleWisePriorAdapter, UpBlock


class TAGEDepth(nn.Module):
    """TAGE-Depth model.

    Inputs:
        rgb:           [B, 3, H, W]
        raw_depth:     [B, 1, H, W]
        mask:          [B, 1, H, W], transparent region mask in {0,1}
        relative_depth:[B, 1, H, W]
        boundary_prior:[B, 1, H, W]
        raw_reliability:[B,1,H,W]

    Outputs:
        pred_depth: predicted transparent-region depth
        final_depth: mask-preserving fused depth
    """

    def __init__(self, base_channels: int = 48, num_scales: int = 4, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = float(max_depth)
        self.num_scales = int(num_scales)

        self.appearance_encoder = EncoderBranch(in_ch=4, base_ch=base_channels, num_scales=num_scales)
        self.geometry_encoder = EncoderBranch(in_ch=2, base_ch=base_channels, num_scales=num_scales)
        self.prior_encoder = EncoderBranch(in_ch=3, base_ch=base_channels, num_scales=num_scales)

        channels = self.appearance_encoder.channels
        self.fuse_layers = nn.ModuleList([
            ConvBlock(ch * 2, ch, stride=1) for ch in channels
        ])
        self.adapters = nn.ModuleList([
            ScaleWisePriorAdapter(feat_ch=ch, prior_ch=ch) for ch in channels
        ])

        rev_channels = list(reversed(channels))
        self.up_blocks = nn.ModuleList()
        for i in range(len(rev_channels) - 1):
            self.up_blocks.append(UpBlock(rev_channels[i], rev_channels[i + 1], rev_channels[i + 1]))

        self.refine_head = nn.Sequential(
            ConvBlock(channels[0], channels[0], stride=1),
            nn.Conv2d(channels[0], 1, kernel_size=3, padding=1),
            nn.Softplus(beta=1.0),
        )

    @staticmethod
    def _cat(*xs):
        return torch.cat(xs, dim=1)

    def forward(
        self,
        rgb: torch.Tensor,
        raw_depth: torch.Tensor,
        mask: torch.Tensor,
        relative_depth: torch.Tensor,
        boundary_prior: torch.Tensor,
        raw_reliability: torch.Tensor,
        reference_depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mask = mask.float().clamp(0.0, 1.0)
        raw_depth = raw_depth.float()
        relative_depth = relative_depth.float()
        boundary_prior = boundary_prior.float()
        raw_reliability = raw_reliability.float()

        app_input = self._cat(rgb.float(), mask)
        geo_input = self._cat(raw_depth, mask)
        prior_input = self._cat(relative_depth, boundary_prior, raw_reliability)

        app_feats = self.appearance_encoder(app_input)
        geo_feats = self.geometry_encoder(geo_input)
        prior_feats = self.prior_encoder(prior_input)

        adapted = []
        for a, g, p, fuse, adapter in zip(app_feats, geo_feats, prior_feats, self.fuse_layers, self.adapters):
            f = fuse(torch.cat([a, g], dim=1))
            adapted.append(adapter(f, p))

        x = adapted[-1]
        skips = list(reversed(adapted[:-1]))
        for up, skip in zip(self.up_blocks, skips):
            x = up(x, skip)

        pred_depth = self.refine_head(x).clamp(0.0, self.max_depth)
        if pred_depth.shape[-2:] != raw_depth.shape[-2:]:
            pred_depth = F.interpolate(pred_depth, size=raw_depth.shape[-2:], mode='bilinear', align_corners=False)
        final_depth = mask * pred_depth + (1.0 - mask) * raw_depth

        out = {
            'pred_depth': pred_depth,
            'final_depth': final_depth,
            'mask': mask,
        }
        if reference_depth is not None:
            out['reference_depth'] = reference_depth
        return out
