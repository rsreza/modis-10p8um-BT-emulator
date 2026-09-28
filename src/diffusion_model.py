"""
diffusion_model.py
==================

A conditional DDPM for generating MODIS 10.8 um brightness temperature
given ICON + SEVIRI conditioning fields.

Uses Hugging Face diffusers:
    - UNet2DConditionModel  : denoiser
    - DDPMScheduler         : training noise schedule
    - DDIMScheduler         : fast deterministic sampling

Conditioning is done by BOTH:
    1. CHANNEL CONCATENATION (spatially resolved):
           input = concat([noisy_target, cond], dim=1)  -> (B, 9, H, W)
    2. CROSS-ATTENTION (global context):
           encoder_hidden_states = pooled(cond)         -> (B, 1, C)

Design goals
------------
- Small enough to train on CPU in a few minutes.
- Clean interface: forward() returns the loss; sample() returns images.
- GPU-ready.

Usage
-----
    from src.diffusion_model import ModisDiffusion

    model = ModisDiffusion()
    loss = model.training_step(cond, target)
    samples = model.sample(cond, n_steps=50)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    in_channels_cond: int = 8
    target_channels: int = 1
    sample_size: int = 64
    block_out_channels: tuple = (32, 64, 96, 128)
    layers_per_block: int = 2
    num_train_timesteps: int = 1000
    num_inference_steps: int = 50
    beta_schedule: str = "squaredcos_cap_v2"
    learning_rate: float = 1e-4
    # Cross-attention context dim: pooled cond channels
    cross_attention_dim: int = 8

    @property
    def unet_in_channels(self) -> int:
        return self.in_channels_cond + self.target_channels


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class ModisDiffusion(nn.Module):
    """
    Wrapper around UNet2DConditionModel with:
        - training_step(cond, target) -> scalar loss
        - sample(cond, n_steps)        -> generated target
    """

    def __init__(self, cfg: DiffusionConfig | None = None, device: str | None = None):
        super().__init__()
        self.cfg = cfg or DiffusionConfig()
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ---- U-Net denoiser ----
        # cross_attention_dim is set so the model has cross-attention layers
        # that accept our pooled conditioning vector.
        self.unet = UNet2DConditionModel(
            sample_size=self.cfg.sample_size,
            in_channels=self.cfg.unet_in_channels,
            out_channels=self.cfg.target_channels,
            layers_per_block=self.cfg.layers_per_block,
            block_out_channels=self.cfg.block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=8,
            cross_attention_dim=self.cfg.cross_attention_dim,
        ).to(self.device)

        # ---- Noise schedulers ----
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg.num_train_timesteps,
            beta_schedule=self.cfg.beta_schedule,
            prediction_type="epsilon",
        )
        self.infer_scheduler = DDIMScheduler(
            num_train_timesteps=self.cfg.num_train_timesteps,
            beta_schedule=self.cfg.beta_schedule,
            prediction_type="epsilon",
            clip_sample=True,
        )
        self.infer_scheduler.set_timesteps(self.cfg.num_inference_steps)

        # ---- Optimizer ----
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=1e-4,
        )

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def _encode_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Build the encoder_hidden_states for cross-attention.

        We use the spatial mean of each conditioning channel, giving a
        (B, 1, C_cond) global context vector.
        """
        # cond: (B, C, H, W) -> (B, C) -> (B, 1, C)
        pooled = cond.mean(dim=(2, 3))                      # (B, C)
        return pooled.unsqueeze(1)                          # (B, 1, C)

    # ------------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------------

    def training_step(
        self,
        cond: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        cond = cond.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        B = target.shape[0]

        t = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps,
            (B,), device=self.device, dtype=torch.long,
        )

        noise = torch.randn_like(target)
        noisy_target = self.train_scheduler.add_noise(target, noise, t)

        unet_input = torch.cat([noisy_target, cond], dim=1)
        encoder_hidden_states = self._encode_cond(cond)

        noise_pred = self.unet(
            unet_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        loss = F.mse_loss(noise_pred, noise)
        return loss

    # ------------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        n_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        cond = cond.to(self.device, non_blocking=True)
        B = cond.shape[0]

        if n_steps is not None and n_steps != self.cfg.num_inference_steps:
            scheduler = DDIMScheduler.from_config(self.infer_scheduler.config)
            scheduler.set_timesteps(n_steps)
        else:
            scheduler = self.infer_scheduler

        sample = torch.randn(
            (B, self.cfg.target_channels,
             self.cfg.sample_size, self.cfg.sample_size),
            device=self.device,
            generator=generator,
        )

        encoder_hidden_states = self._encode_cond(cond)

        for t in scheduler.timesteps:
            unet_input = torch.cat([sample, cond], dim=1)
            t_batch = t.expand(B) if t.dim() == 0 else t
            noise_pred = self.unet(
                unet_input,
                t_batch,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            sample = scheduler.step(noise_pred, t, sample).prev_sample

        return sample

    # ------------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "unet": self.unet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(ckpt["unet"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building model ...")
    model = ModisDiffusion()
    n_params = sum(p.numel() for p in model.unet.parameters())
    print(f"  device         : {model.device}")
    print(f"  U-Net params   : {n_params/1e6:.2f} M")
    print(f"  input channels : {model.cfg.unet_in_channels}")
    print(f"  sample size    : {model.cfg.sample_size}")

    B, C_cond, H, W = 2, model.cfg.in_channels_cond, 64, 64
    cond = torch.randn(B, C_cond, H, W)
    target = torch.randn(B, 1, H, W)

    print("\nTraining step ...")
    loss = model.training_step(cond, target)
    print(f"  loss = {loss.item():.4f}")

    print("\nSampling ...")
    samples = model.sample(cond, n_steps=10)
    print(f"  samples shape = {tuple(samples.shape)}")
    print(f"  samples range = [{samples.min():.2f}, {samples.max():.2f}]")

    print("\nAll model sanity checks passed.")
