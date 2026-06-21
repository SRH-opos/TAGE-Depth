import torch
import torch.nn.functional as F


def build_boundary_prior(mask: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Build mask-boundary prior from a binary transparent mask."""
    pad = kernel_size // 2
    mask = mask.float().clamp(0.0, 1.0)
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def robust_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b = x.shape[0]
    flat = x.reshape(b, -1)
    lo = flat.quantile(0.02, dim=1).view(b, 1, 1, 1)
    hi = flat.quantile(0.98, dim=1).view(b, 1, 1, 1)
    return ((x - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


def build_raw_reliability(raw_depth: torch.Tensor, relative_depth: torch.Tensor) -> torch.Tensor:
    """Build raw-reliability prior from raw-relative discrepancy."""
    diff = torch.abs(raw_depth.float() - relative_depth.float())
    return 1.0 - robust_norm(diff)
