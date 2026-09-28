"""
evaluate.py
===========

Evaluate the trained conditional diffusion emulator.

What it does
------------
1. Loads a trained checkpoint.
2. Samples MODIS-like BT images conditioned on held-out ICON + SEVIRI inputs.
3. Denormalizes to physical Kelvin.
4. Computes metrics against the true MODIS BT:
       - RMSE
       - Bias (mean signed error)
       - MAE
       - Wasserstein distance (distributional similarity)
       - Correlation
       - Uncertainty (std across multiple stochastic samples)
5. Produces side-by-side comparison figures:
       true MODIS | generated | difference
6. Saves everything to results/figures and results/metrics.

Usage
-----
    # Use the latest checkpoint
    python -m src.evaluate

    # Use a specific checkpoint
    python -m src.evaluate --checkpoint checkpoints/model_ep005.pt

    # More stochastic samples for uncertainty quantification
    python -m src.evaluate --n-samples 8
"""

# ============================================================================
# Threading — MUST BE SET BEFORE numpy / torch / scipy imports
# ============================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from scipy.stats import wasserstein_distance

from src.dataset import ModisIconDataset, NORM_STATS
from src.diffusion_model import DiffusionConfig, ModisDiffusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalize_bt(x: torch.Tensor) -> torch.Tensor:
    """Convert normalized MODIS BT back to Kelvin."""
    mean, std = NORM_STATS["modis_bt"]
    return x * std + mean


def compute_metrics(pred_k: np.ndarray, true_k: np.ndarray) -> dict:
    """Compute RMSE, bias, MAE, Wasserstein, correlation."""
    diff = pred_k - true_k
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))

    # Wasserstein distance on flattened pixels
    wd = float(wasserstein_distance(pred_k.ravel(), true_k.ravel()))

    # Correlation
    p = pred_k.ravel()
    t = true_k.ravel()
    if p.std() > 0 and t.std() > 0:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = float("nan")

    return {
        "rmse": rmse,
        "bias": bias,
        "mae": mae,
        "wasserstein": wd,
        "correlation": corr,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint: Path,
    n_samples: int,
    n_inference_steps: int,
    tile_size: int,
    seed: int,
    out_figures: Path,
    out_metrics: Path,
) -> None:
    print("=" * 70)
    print("Conditional diffusion emulator — evaluation")
    print("=" * 70)
    print(f"  checkpoint       : {checkpoint}")
    print(f"  n_samples        : {n_samples}")
    print(f"  n_inference_steps: {n_inference_steps}")
    print(f"  tile_size        : {tile_size}")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ---- Load dataset (small samples_per_epoch just to have access) ----
    print("\nLoading dataset ...")
    ds = ModisIconDataset(tile_size=tile_size, samples_per_epoch=1, seed=seed)

    # ---- Load model ----
    cfg = DiffusionConfig(
        in_channels_cond=ds.cond.shape[1],
        target_channels=1,
        sample_size=tile_size,
        num_inference_steps=n_inference_steps,
    )
    model = ModisDiffusion(cfg=cfg)
    print(f"Loading checkpoint {checkpoint} ...")
    model.load(str(checkpoint))
    model.unet.eval()
    print(f"  device: {model.device}")

    # ---- Pick a fixed test tile ----
    # Use the first pass (t=0), top-left tile.
    ip = 0
    i = 20
    j = 100
    cond_full = ds.cond[ip:ip+1, :, i:i+tile_size, j:j+tile_size]
    true_full = ds.target[ip:ip+1, :, i:i+tile_size, j:j+tile_size]

    print(f"\nTest tile: pass={ip}, row={i}, col={j}")
    print(f"  cond shape: {tuple(cond_full.shape)}")
    print(f"  true shape: {tuple(true_full.shape)}")

    # ---- Generate n_samples stochastic samples ----
    print(f"\nGenerating {n_samples} samples (DDIM {n_inference_steps} steps) ...")
    samples_k = np.empty((n_samples, tile_size, tile_size), dtype=np.float32)

    with torch.no_grad():
        for k in range(n_samples):
            gen = torch.Generator(device=model.device).manual_seed(seed + k)
            s = model.sample(
                cond_full, n_steps=n_inference_steps, generator=gen,
            )
            samples_k[k] = denormalize_bt(s)[0, 0].cpu().numpy()

    # Ensemble mean and std
    ensemble_mean = samples_k.mean(axis=0)
    ensemble_std = samples_k.std(axis=0)

    # True (denormalized)
    true_k = denormalize_bt(true_full)[0, 0].cpu().numpy()

    # ---- Metrics ----
    print("\nComputing metrics ...")
    metrics_mean = compute_metrics(ensemble_mean, true_k)
    metrics_first = compute_metrics(samples_k[0], true_k)

    print("\n  --- Ensemble mean vs truth ---")
    for k, v in metrics_mean.items():
        print(f"    {k:14s}: {v:+.4f}")

    print("\n  --- Single sample (k=0) vs truth ---")
    for k, v in metrics_first.items():
        print(f"    {k:14s}: {v:+.4f}")

    # ---- Save metrics ----
    out_metrics.mkdir(parents=True, exist_ok=True)
    metrics_path = out_metrics / "evaluation.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "checkpoint": str(checkpoint),
            "test_tile": {"pass": ip, "row": i, "col": j},
            "n_samples": n_samples,
            "n_inference_steps": n_inference_steps,
            "ensemble_mean_vs_true": metrics_mean,
            "single_sample_vs_true": metrics_first,
        }, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # ---- Figures ----
    out_figures.mkdir(parents=True, exist_ok=True)

    # Figure 1: true | generated | difference
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin = min(true_k.min(), ensemble_mean.min())
    vmax = max(true_k.max(), ensemble_mean.max())

    im0 = axes[0].imshow(true_k, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0].set_title("True MODIS BT")
    plt.colorbar(im0, ax=axes[0], label="K")

    im1 = axes[1].imshow(ensemble_mean, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Generated (ensemble of {n_samples})")
    plt.colorbar(im1, ax=axes[1], label="K")

    diff = ensemble_mean - true_k
    dmax = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title("Generated − True")
    plt.colorbar(im2, ax=axes[2], label="K")

    for ax in axes:
        ax.set_xlabel("lon idx")
        ax.set_ylabel("lat idx")

    plt.tight_layout()
    fig.savefig(out_figures / "true_vs_generated.png", dpi=120)
    plt.close(fig)

    # Figure 2: uncertainty (ensemble std) + sample spread
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(true_k, cmap="turbo")
    axes[0].set_title("True")
    plt.colorbar(im0, ax=axes[0], label="K")

    im1 = axes[1].imshow(ensemble_std, cmap="magma")
    axes[1].set_title("Ensemble std (uncertainty)")
    plt.colorbar(im1, ax=axes[1], label="K")

    im2 = axes[2].imshow(samples_k[0], cmap="turbo")
    axes[2].set_title("Single sample (k=0)")
    plt.colorbar(im2, ax=axes[2], label="K")

    plt.tight_layout()
    fig.savefig(out_figures / "uncertainty.png", dpi=120)
    plt.close(fig)

    # Figure 3: histogram of predicted vs true BT (distribution check)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(true_k.min(), samples_k.min()),
                       max(true_k.max(), samples_k.max()), 40)
    ax.hist(true_k.ravel(), bins=bins, alpha=0.5, label="True", density=True)
    ax.hist(ensemble_mean.ravel(), bins=bins, alpha=0.5, label="Generated", density=True)
    ax.set_xlabel("BT (K)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of BT — true vs generated")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_figures / "distribution.png", dpi=120)
    plt.close(fig)

    print(f"Figures saved in: {out_figures}")
    print("  - true_vs_generated.png")
    print("  - uncertainty.png")
    print("  - distribution.png")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(dir_: Path) -> Path:
    ckpts = sorted(dir_.glob("model_ep*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {dir_}")
    return ckpts[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate diffusion emulator")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to model checkpoint (default: latest)")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Number of stochastic samples (for uncertainty)")
    parser.add_argument("--n-inference-steps", type=int, default=50,
                        help="DDIM inference steps")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-figures", type=Path,
                        default=Path("results/figures"))
    parser.add_argument("--out-metrics", type=Path,
                        default=Path("results/metrics"))
    args = parser.parse_args()

    ckpt = args.checkpoint or _find_latest_checkpoint(Path("checkpoints"))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    evaluate(
        checkpoint=ckpt,
        n_samples=args.n_samples,
        n_inference_steps=args.n_inference_steps,
        tile_size=args.tile_size,
        seed=args.seed,
        out_figures=args.out_figures,
        out_metrics=args.out_metrics,
    )


if __name__ == "__main__":
    main()
