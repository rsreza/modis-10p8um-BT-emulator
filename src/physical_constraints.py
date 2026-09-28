"""
physical_constraints.py
=======================

Differentiable physical-consistency terms that can be added to the diffusion
training loss, plus hard post-processing functions that can be applied to
generated samples.

Four independent constraints
----------------------------
1. BT bounds           — quadratic hinge outside [lo, hi] Kelvin
2. Spatial smoothness  — penalize high-frequency noise via Laplacian
3. Temporal coherence  — penalize abrupt jumps between consecutive frames
4. Cloud/BT consistency — penalize when the model predicts cold BT outside
                          the cloud mask and warm BT inside it
                          (uses SEVIRI channel or ICON cloud info if available)

Each is a scalar tensor suitable for addition to any loss:
    loss_total = loss_diffusion
               + λ_bounds * bounds_penalty(bt_kelvin)
               + λ_smooth * spatial_smoothness_penalty(bt_kelvin)
               + λ_temp   * temporal_smoothness_penalty(bt_sequence_kelvin)

Design
------
- Pure functions, no state.
- All ops use torch so gradients flow where it matters.
- Hard clipping helper provided for inference-time post-processing.

Usage
-----
    from src.physical_constraints import (
        bt_bounds_penalty,
        spatial_smoothness_penalty,
        temporal_smoothness_penalty,
        cloud_bt_consistency_penalty,
        clip_bt,
        combined_physical_loss,
    )

    bt_k = sample * 15.0 + 270.0     # denormalize
    loss = loss_diffusion \
         + 0.01 * bt_bounds_penalty(bt_k) \
         + 0.001 * spatial_smoothness_penalty(bt_k)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BT_MIN_K_DEFAULT: float = 180.0
BT_MAX_K_DEFAULT: float = 320.0


# ---------------------------------------------------------------------------
# 1. Brightness-temperature bounds
# ---------------------------------------------------------------------------

def bt_bounds_penalty(
    bt_kelvin: torch.Tensor,
    lo: float = BT_MIN_K_DEFAULT,
    hi: float = BT_MAX_K_DEFAULT,
) -> torch.Tensor:
    """
    Quadratic hinge penalty for BT outside [lo, hi].

    Input shape: any tensor of BT values in Kelvin.
    Returns: scalar tensor (mean of squared violations).
    """
    below = torch.clamp(lo - bt_kelvin, min=0.0)
    above = torch.clamp(bt_kelvin - hi, min=0.0)
    return (below ** 2 + above ** 2).mean()


# ---------------------------------------------------------------------------
# 2. Spatial smoothness (Laplacian)
# ---------------------------------------------------------------------------

def _laplacian_kernel(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """5-point Laplacian filter for 2-D images."""
    k = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=dtype, device=device,
    )
    return k.view(1, 1, 3, 3)


def spatial_smoothness_penalty(
    bt_kelvin: torch.Tensor,
    weight_by_gradient: bool = False,
) -> torch.Tensor:
    """
    Penalize high-frequency spatial structure.

    Parameters
    ----------
    bt_kelvin : (B, 1, H, W) — BT in Kelvin
    weight_by_gradient : if True, down-weight smoothing near strong gradients
                         (cloud edges), avoiding unrealistically soft transitions.

    Returns
    -------
    Scalar tensor (mean squared Laplacian).
    """
    if bt_kelvin.dim() != 4:
        raise ValueError(f"expected (B, 1, H, W), got {tuple(bt_kelvin.shape)}")

    kernel = _laplacian_kernel(bt_kelvin.dtype, bt_kelvin.device)
    lap = F.conv2d(bt_kelvin, kernel, padding=1)

    if weight_by_gradient:
        # Local gradient magnitude (Sobel-like)
        gx = torch.abs(bt_kelvin[:, :, :, 1:] - bt_kelvin[:, :, :, :-1])
        gy = torch.abs(bt_kelvin[:, :, 1:, :] - bt_kelvin[:, :, :-1, :])
        # Match shape of lap
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        w = 1.0 / (1.0 + gx + gy)
        return (w * lap ** 2).mean()

    return (lap ** 2).mean()


# ---------------------------------------------------------------------------
# 3. Temporal smoothness (between consecutive frames)
# ---------------------------------------------------------------------------

def temporal_smoothness_penalty(bt_sequence_kelvin: torch.Tensor) -> torch.Tensor:
    """
    Penalize abrupt frame-to-frame jumps in a BT time series.

    Parameters
    ----------
    bt_sequence_kelvin : (B, T, 1, H, W) or (T, 1, H, W)

    Returns
    -------
    Scalar tensor.
    """
    if bt_sequence_kelvin.dim() == 5:
        # (B, T, 1, H, W) -> diff along T
        diffs = bt_sequence_kelvin[:, 1:] - bt_sequence_kelvin[:, :-1]
    elif bt_sequence_kelvin.dim() == 4:
        # (T, 1, H, W) -> treat as single sequence
        diffs = bt_sequence_kelvin[1:] - bt_sequence_kelvin[:-1]
    else:
        raise ValueError(f"expected (B,T,1,H,W) or (T,1,H,W), got "
                         f"{tuple(bt_sequence_kelvin.shape)}")
    return (diffs ** 2).mean()


# ---------------------------------------------------------------------------
# 4. Cloud / BT consistency
# ---------------------------------------------------------------------------

def cloud_bt_consistency_penalty(
    bt_kelvin: torch.Tensor,
    cloud_mask: torch.Tensor,
    clear_bt_min_k: float = 275.0,
    cloudy_bt_max_k: float = 265.0,
) -> torch.Tensor:
    """
    Penalize physically implausible combinations of cloud mask and BT.

    - Clear pixel with cold BT  -> penalty (should be warm)
    - Cloudy pixel with warm BT -> penalty (should be cold)

    Parameters
    ----------
    bt_kelvin   : (B, 1, H, W)
    cloud_mask  : (B, 1, H, W)  in {0, 1} (soft values 0..1 also accepted)
    clear_bt_min_k : clear pixels should be >= this
    cloudy_bt_max_k: cloudy pixels should be <= this

    Returns
    -------
    Scalar tensor.
    """
    mask = cloud_mask.to(bt_kelvin.dtype)
    clear = 1.0 - mask
    cloudy = mask

    # Clear pixel that is too cold
    clear_cold = torch.clamp(clear_bt_min_k - bt_kelvin, min=0.0) ** 2
    # Cloudy pixel that is too warm
    cloudy_warm = torch.clamp(bt_kelvin - cloudy_bt_max_k, min=0.0) ** 2

    return (clear * clear_cold + cloudy * cloudy_warm).mean()


# ---------------------------------------------------------------------------
# 5. Hard post-processing (inference time)
# ---------------------------------------------------------------------------

def clip_bt(
    bt_kelvin: torch.Tensor,
    lo: float = BT_MIN_K_DEFAULT,
    hi: float = BT_MAX_K_DEFAULT,
) -> torch.Tensor:
    """Hard-clip BT to the physically plausible range."""
    return torch.clamp(bt_kelvin, lo, hi)


# ---------------------------------------------------------------------------
# 6. Combined helper
# ---------------------------------------------------------------------------

def combined_physical_loss(
    bt_pred_kelvin: torch.Tensor,
    bt_sequence_kelvin: Optional[torch.Tensor] = None,
    cloud_mask: Optional[torch.Tensor] = None,
    lambda_bounds: float = 1.0,
    lambda_smooth: float = 0.1,
    lambda_temporal: float = 0.0,
    lambda_cloud: float = 0.0,
) -> dict:
    """
    Return individual + total physical penalty.

    Returns
    -------
    dict with keys:
        'bounds'   : scalar tensor
        'smooth'   : scalar tensor
        'temporal' : scalar tensor or None
        'cloud'    : scalar tensor or None
        'total'    : scalar tensor (weighted sum)
    """
    terms: dict = {}

    terms["bounds"] = bt_bounds_penalty(bt_pred_kelvin)
    terms["smooth"] = spatial_smoothness_penalty(bt_pred_kelvin)

    if bt_sequence_kelvin is not None and lambda_temporal > 0.0:
        terms["temporal"] = temporal_smoothness_penalty(bt_sequence_kelvin)
    else:
        terms["temporal"] = None

    if cloud_mask is not None and lambda_cloud > 0.0:
        terms["cloud"] = cloud_bt_consistency_penalty(
            bt_pred_kelvin, cloud_mask
        )
    else:
        terms["cloud"] = None

    total = (
        lambda_bounds * terms["bounds"]
        + lambda_smooth * terms["smooth"]
    )
    if terms["temporal"] is not None:
        total = total + lambda_temporal * terms["temporal"]
    if terms["cloud"] is not None:
        total = total + lambda_cloud * terms["cloud"]

    terms["total"] = total
    return terms


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing physical_constraints ...")

    # Normal BT
    bt_ok = torch.full((2, 1, 32, 32), 280.0)
    # Contains out-of-bounds
    bt_bad = bt_ok.clone()
    bt_bad[0, 0, 0, 0] = 400.0    # too hot
    bt_bad[1, 0, 5, 5] = 100.0    # too cold

    print(f"  bounds penalty (ok)  = {bt_bounds_penalty(bt_ok).item():.4f}")
    print(f"  bounds penalty (bad) = {bt_bounds_penalty(bt_bad).item():.4f}")

    # Spatial smoothness
    bt_smooth = torch.full((1, 1, 32, 32), 280.0)
    bt_noisy = bt_smooth + 10.0 * torch.randn(1, 1, 32, 32)
    print(f"  smooth penalty (smooth) = {spatial_smoothness_penalty(bt_smooth).item():.4f}")
    print(f"  smooth penalty (noisy)  = {spatial_smoothness_penalty(bt_noisy).item():.4f}")

    # Temporal
    bt_seq_ok = torch.full((4, 1, 32, 32), 280.0)
    bt_seq_jumpy = bt_seq_ok.clone()
    bt_seq_jumpy[2] += 50.0
    print(f"  temporal penalty (ok)    = {temporal_smoothness_penalty(bt_seq_ok).item():.4f}")
    print(f"  temporal penalty (jumpy) = {temporal_smoothness_penalty(bt_seq_jumpy).item():.4f}")

    # Cloud/BT consistency
    mask = torch.zeros((1, 1, 32, 32))
    mask[0, 0, :16, :] = 1.0     # top half cloudy
    bt_consistent = torch.full((1, 1, 32, 32), 280.0)
    bt_consistent[0, 0, :16, :] = 250.0   # cold where cloudy
    bt_inconsistent = bt_consistent.flip(0)
    # flip did nothing since batch=1; make explicitly wrong:
    bt_wrong = torch.full((1, 1, 32, 32), 280.0)   # all warm, but top half cloudy
    print(f"  cloud penalty (ok)    = {cloud_bt_consistency_penalty(bt_consistent, mask).item():.4f}")
    print(f"  cloud penalty (wrong) = {cloud_bt_consistency_penalty(bt_wrong, mask).item():.4f}")

    # Combined
    out = combined_physical_loss(
        bt_bad,
        bt_sequence_kelvin=bt_seq_jumpy.unsqueeze(0),
        cloud_mask=mask,
        lambda_bounds=1.0,
        lambda_smooth=0.1,
        lambda_temporal=0.01,
        lambda_cloud=0.1,
    )
    print("\nCombined terms:")
    for k, v in out.items():
        if v is None:
            print(f"  {k:10s}: None")
        else:
            print(f"  {k:10s}: {v.item():.6f}")

    # Hard clip
    bt_clipped = clip_bt(bt_bad)
    print(f"\nHard clip: min={bt_clipped.min().item():.1f}  max={bt_clipped.max().item():.1f}")
    assert bt_clipped.min() >= 180.0 and bt_clipped.max() <= 320.0

    print("\nAll physical_constraints sanity checks passed.")
