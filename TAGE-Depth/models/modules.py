import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution-BN-ReLU block used by the lightweight release backbone."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBranch(nn.Module):
    """Multi-scale encoder branch."""

    def __init__(self, in_ch: int, base_ch: int = 48, num_scales: int = 4):
        super().__init__()
        channels = [base_ch * (2 ** i) for i in range(num_scales)]
        layers = []
        prev = in_ch
        for i, ch in enumerate(channels):
            layers.append(ConvBlock(prev, ch, stride=1 if i == 0 else 2))
            prev = ch
        self.layers = nn.ModuleList(layers)
        self.channels = channels

    def forward(self, x: torch.Tensor):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class ScaleWisePriorAdapter(nn.Module):
    """Scale-wise prior adapter.

    It receives fused appearance-geometry features and prior features at the
    same scale, then generates a residual response controlled by a modulation map.
    """

    def __init__(self, feat_ch: int, prior_ch: int):
        super().__init__()
        self.prior_align = nn.Sequential(
            nn.Conv2d(prior_ch, feat_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.response = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1, bias=True),
        )
        self.modulation = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, fused_feat: torch.Tensor, prior_feat: torch.Tensor) -> torch.Tensor:
        if prior_feat.shape[-2:] != fused_feat.shape[-2:]:
            prior_feat = F.interpolate(prior_feat, size=fused_feat.shape[-2:], mode='bilinear', align_corners=False)
        prior_feat = self.prior_align(prior_feat)
        joint = torch.cat([fused_feat, prior_feat], dim=1)
        residual = self.response(joint)
        alpha = self.modulation(joint)
        return fused_feat + alpha * residual


class UpBlock(nn.Module):
    """Upsampling block for decoder aggregation."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
