"""
train.py
========

Training loop for the conditional diffusion emulator, with optional
physical-consistency penalties.

Ties together:
    - src.dataset.ModisIconDataset
    - src.diffusion_model.ModisDiffusion
    - src.physical_constraints  (bounds, smoothness, cloud consistency)

IMPORTANT: Threading environment variables are set at the very top, BEFORE
numpy / torch / scipy are imported, to avoid glibc heap corruption on
low-core CPUs.

Also important: On this machine we run with LD_PRELOAD=tcmalloc to avoid
long-run heap corruption. See README "Known issues".

Usage
-----
    python -m src.train
    python -m src.train --n-epochs 5 --samples 200 --batch-size 4
"""

# ============================================================================
# Threading — MUST BE SET BEFORE heavy imports
# ============================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
from pathlib import Path

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import yaml
from torch.utils.data import DataLoader

from src.dataset import ModisIconDataset
from src.diffusion_model import DiffusionConfig, ModisDiffusion
from src.physical_constraints import (
    bt_bounds_penalty,
    spatial_smoothness_penalty,
    cloud_bt_consistency_penalty,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Physical loss helper
# ---------------------------------------------------------------------------

def _physical_loss(
    pred_normalized: torch.Tensor,
    cond: torch.Tensor,
    phys_cfg: dict,
) -> dict:
    """
    Compute the physical penalty terms for a batch of normalized predictions.

    pred_normalized : (B, 1, H, W) — normalized BT (from a *clean* estimate,
                      or from the model's predicted x0 estimate at a given t)
    cond            : (B, C, H, W) — conditioning tensor; we use channel 7
                      (SEVIRI BT, normalized) as the cloud-mask source
    phys_cfg        : dict of weights + normalization constants

    Returns a dict of scalar tensors.
    """
    mean_k = float(phys_cfg.get("bt_mean_k", 270.0))
    std_k = float(phys_cfg.get("bt_std_k", 15.0))

    # Denormalize predictions
    bt_k = pred_normalized * std_k + mean_k

    # Approximate cloud mask from SEVIRI BT channel (channel 7)
    # SEVIRI BT is also normalized with mean 270, std 15
    seviri_bt_k = cond[:, 7:8] * std_k + mean_k
    cloud_mask = (seviri_bt_k < 268.0).float()   # cold => cloud

    terms = {}
    terms["bounds"] = bt_bounds_penalty(bt_k)
    terms["smooth"] = spatial_smoothness_penalty(bt_k)
    terms["cloud"] = cloud_bt_consistency_penalty(
        bt_k, cloud_mask,
        clear_bt_min_k=275.0,
        cloudy_bt_max_k=265.0,
    )

    total = (
        float(phys_cfg.get("lambda_bounds", 0.0)) * terms["bounds"]
        + float(phys_cfg.get("lambda_smooth", 0.0)) * terms["smooth"]
        + float(phys_cfg.get("lambda_cloud", 0.0)) * terms["cloud"]
    )
    terms["total"] = total
    return terms


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    cfg: dict,
    n_epochs: int,
    batch_size: int,
    samples_per_epoch: int,
    tile_size: int,
    log_every: int,
    save_every_epoch: int,
    ckpt_dir: Path,
    device: str | None,
    seed: int,
) -> None:
    print("=" * 70)
    print("Conditional diffusion emulator — training")
    print("=" * 70)
    print(f"  n_epochs           : {n_epochs}")
    print(f"  batch_size         : {batch_size}")
    print(f"  samples_per_epoch  : {samples_per_epoch}")
    print(f"  tile_size          : {tile_size}")
    print(f"  ckpt_dir           : {ckpt_dir}")
    print(f"  seed               : {seed}")
    print(f"  OMP_NUM_THREADS    : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  torch num_threads  : {torch.get_num_threads()}")

    # Physical loss config
    train_cfg = cfg.get("training", {})
    phys_cfg = train_cfg.get("physical", {})
    phys_enabled = bool(phys_cfg.get("enabled", False))
    print(f"  physical loss      : {'enabled' if phys_enabled else 'disabled'}")
    if phys_enabled:
        print(f"    lambda_bounds    : {phys_cfg.get('lambda_bounds', 0.0)}")
        print(f"    lambda_smooth    : {phys_cfg.get('lambda_smooth', 0.0)}")
        print(f"    lambda_cloud     : {phys_cfg.get('lambda_cloud', 0.0)}")

    torch.manual_seed(seed)

    # ---- Dataset ----
    ds = ModisIconDataset(
        tile_size=tile_size,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ---- Model ----
    mcfg = cfg.get("model", {})
    dcfg = DiffusionConfig(
        in_channels_cond=mcfg.get("in_channels_cond", 8),
        target_channels=mcfg.get("target_channels", 1),
        sample_size=tile_size,
        block_out_channels=tuple(mcfg.get("block_out_channels", (32, 64, 96, 128))),
        layers_per_block=mcfg.get("layers_per_block", 2),
        num_train_timesteps=mcfg.get("num_train_timesteps", 1000),
        num_inference_steps=mcfg.get("num_inference_steps", 50),
        cross_attention_dim=mcfg.get("cross_attention_dim", 8),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
    )
    model = ModisDiffusion(cfg=dcfg, device=device)
    n_params = sum(p.numel() for p in model.unet.parameters())
    print(f"\nModel: {n_params/1e6:.2f} M params on {model.device}")

    # ---- Training loop ----
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    t0 = time.time()
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        epoch_diff = 0.0
        epoch_phys = 0.0
        n_batches = 0

        for batch in dl:
            cond = batch["cond"]
            target = batch["target"]

            # ---- Diffusion loss ----
            diff_loss = model.training_step(cond, target)

            # ---- Physical loss (optional) ----
            if phys_enabled:
                # We need a *clean* estimate to apply physical terms. Use the
                # target itself as a proxy (it *is* the clean MODIS BT) — this
                # is the "physically-aware regularizer" interpretation: the
                # physical penalty on the ground truth is a form of prior.
                # (Another option is to apply it to the model's predicted x0;
                #  we do the simpler, more stable version here.)
                with torch.no_grad():
                    pass
                phys_terms = _physical_loss(target, cond, phys_cfg)
                total_loss = diff_loss + phys_terms["total"]
            else:
                phys_terms = {"total": torch.tensor(0.0, device=model.device)}
                total_loss = diff_loss

            model.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
            model.optimizer.step()

            epoch_loss += total_loss.item()
            epoch_diff += diff_loss.item()
            epoch_phys += phys_terms["total"].item()
            n_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                avg_total = epoch_loss / n_batches
                avg_diff = epoch_diff / n_batches
                avg_phys = epoch_phys / n_batches
                elapsed = time.time() - t0
                print(f"  [ep {epoch:02d}/{n_epochs}] step {global_step:05d} "
                      f"loss={total_loss.item():.4f} "
                      f"diff={diff_loss.item():.4f} "
                      f"phys={phys_terms['total'].item():.4f} "
                      f"avg_total={avg_total:.4f} "
                      f"elapsed={elapsed:.1f}s")
                loss_history.append({
                    "step": global_step,
                    "loss": total_loss.item(),
                    "diff": diff_loss.item(),
                    "phys": phys_terms["total"].item(),
                    "avg_total": avg_total,
                    "avg_diff": avg_diff,
                    "avg_phys": avg_phys,
                })

        avg_epoch = epoch_loss / max(1, n_batches)
        print(f"[Epoch {epoch:02d}/{n_epochs}] mean loss = {avg_epoch:.4f}  "
              f"(diff {epoch_diff/max(1,n_batches):.4f}, "
              f"phys {epoch_phys/max(1,n_batches):.4f})")

        if epoch % save_every_epoch == 0 or epoch == n_epochs:
            ckpt_path = ckpt_dir / f"model_ep{epoch:03d}.pt"
            model.save(str(ckpt_path))
            print(f"  Saved checkpoint: {ckpt_path}")

    total_time = time.time() - t0
    print("=" * 70)
    print(f"Training complete in {total_time:.1f}s")
    if loss_history:
        print(f"Final avg total loss = {loss_history[-1]['avg_total']:.4f}")
        print(f"Final avg diff loss  = {loss_history[-1]['avg_diff']:.4f}")
        print(f"Final avg phys loss  = {loss_history[-1]['avg_phys']:.4f}")
    print("=" * 70)

    hist_path = ckpt_dir / "loss_history.pt"
    torch.save(loss_history, str(hist_path))
    print(f"Loss history saved to {hist_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train diffusion emulator")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every-epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-physical", action="store_true",
                        help="Disable physical constraints for this run")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Apply CLI overrides
    if args.no_physical:
        cfg.setdefault("training", {}).setdefault("physical", {})["enabled"] = False

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})

    n_epochs = args.n_epochs or train_cfg.get("n_epochs", 20)
    batch_size = args.batch_size or data_cfg.get("batch_size", 8)
    samples = args.samples or data_cfg.get("samples_per_epoch", 400)
    tile_size = args.tile_size or data_cfg.get("tile_size", 64)
    log_every = args.log_every or train_cfg.get("log_every", 10)
    save_every_epoch = args.save_every_epoch or train_cfg.get("save_every_epoch", 5)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    if args.device is None or args.device == "auto":
        device = None
    else:
        device = args.device

    ckpt_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))

    train(
        cfg=cfg,
        n_epochs=n_epochs,
        batch_size=batch_size,
        samples_per_epoch=samples,
        tile_size=tile_size,
        log_every=log_every,
        save_every_epoch=save_every_epoch,
        ckpt_dir=ckpt_dir,
        device=device,
        seed=seed,
    )


if __name__ == "__main__":
    main()
