# MODIS 10.8 µm BT-emulator at high temporal resolution

### From Weather Forecasts to Satellite-Style Imagery

A **conditional diffusion model** that learns to generate high-resolution
MODIS-like 10.8 µm brightness temperature (BT) fields from **ICON**
numerical weather prediction data, constrained by temporally dense
**SEVIRI** observations.

<p align="center">
  <img src="docs/methodology.png"
       alt="Methodology overview"
       width="100%">
</p>

<p align="center">
  <em>Methodology: three synthetic data sources feed a conditional diffusion
  model that generates MODIS-resolution brightness temperature fields,
  evaluated with physical metrics and uncertainty quantification.</em>
</p>

---

## Overview

MODIS (polar-orbiting) provides high spatial resolution (1 km) but only a
couple of overpasses per day. SEVIRI (geostationary) provides continuous
coverage (every 15 min) but at coarser resolution (3 km). ICON provides the
underlying physics — but not the imagery. This project builds a **diffusion
emulator** that fuses all three: **given an ICON forecast and a SEVIRI
observation, generate the MODIS image that a MODIS overpass would have
captured.**

This repository is a **proof-of-concept**, developed entirely on synthetic
data whose statistics mimic the real operational products. The same code
runs on real data with only path updates.

---

## Table of contents

- [Methodology](#methodology)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Data description](#data-description)
- [Model description](#model-description)
- [Physical constraints](#physical-constraints)
- [Results](#results)
- [Configuration](#configuration)
- [Known issues](#known-issues)
- [Swapping in real data](#swapping-in-real-data)
- [Limitations](#limitations)
- [Future work](#future-work)
- [License](#license)

---

## Methodology

The pipeline has **three stages** (see figure above):

### 1 · Synthetic Data Generation

**1.1 — Generate ICON fields.**
Physically plausible vertical profiles: moist-adiabatic lapse rate
(≈ 6.5 K/km up to 11 km), exponential humidity decay (scale height 2 km),
near-surface nighttime inversion, diurnal surface-temperature cycle.
Clouds are Gaussian blobs advected eastward at 10 m/s, giving temporal
continuity across the 8-hour forecast window.

**1.2 — Derive MODIS brightness temperature.**
Simplified radiative transfer:

- **Clear sky:** `BT = Ts − γ·W + offset`, with `γ = 0.3 K/mm` and
  `offset = 3 K`.
- **Cloudy:** `BT = T(z_top) + ε`, where `ε` is small additive noise.

Output at 1 km grid, at the actual overpass times (00, 06, 12, 18 UTC).

**1.3 — Derive SEVIRI brightness temperature.**
Area-average the MODIS field to 3 km; linearly interpolate between overpasses
to fill all 33 time steps; add sensor and temporal noise. Result mimics
SEVIRI's high cadence and coarse resolution.

### 2 · Conditional Diffusion Model

**2.1 — Prepare training data.**
All sources regridded to the MODIS 1 km grid. Eight conditioning channels
stacked (7 ICON + 1 SEVIRI):

| # | Channel | Source |
|---|---|---|
| 0 | Surface temperature `Ts` | ICON |
| 1 | Surface-level temperature | ICON |
| 2 | Surface-level humidity | ICON |
| 3 | Column-max cloud water `Lc` | ICON |
| 4 | Column-max cloud ice `Ic` | ICON |
| 5 | Column-max effective radius `re` | ICON |
| 6 | Cloud-top height `z_top` | ICON |
| 7 | SEVIRI 10.8 µm BT | SEVIRI |

Training samples are random 64 × 64 crops (CPU-friendly).

**2.2 — Train a conditional DDPM.**
U-Net denoiser (4 down / 4 up blocks, ~5.8 M parameters), conditioned by:

- **channel concatenation** at the U-Net input, and
- **cross-attention** on a mean-pooled version of the conditioning vector.

Training minimizes the standard ε-prediction MSE, optionally augmented with
physical penalties (see below). Training uses a DDPM schedule with 1000
steps; inference uses DDIM with 50 steps.

### 3 · Evaluation & Results

**3.1 — Sampling.** Generate MODIS-like images for held-out inputs.

**3.2 — Metrics.** RMSE, bias, MAE, Wasserstein distance, correlation.

**3.3 — Uncertainty.** Multiple stochastic samples per conditioning input
give an ensemble spread that reveals where the model is uncertain
(typically near cloud edges).

**3.4 — Figures.** True vs generated, difference maps, uncertainty maps,
distribution histograms.

---

## Quick start

### Requirements

- Python ≥ 3.10
- Linux / macOS (Windows untested)
- ~2 GB free disk for dependencies, ~1 GB for generated data
- ≥ 8 GB RAM recommended (works with 4 GB with smaller batch sizes)

### Install

```bash
git clone https://github.com/rsreza/modis-10p8um-BT-emulator.git
cd modis-10p8um-BT-emulator

python3 -m venv .venv
source .venv/bin/activate

# CPU-only (works everywhere):
pip install -r requirements.txt

# GPU (CUDA 12.1) alternative:
# pip install -r requirements-gpu.txt
```

### Run the pipeline

```bash
# 1. Generate synthetic ICON, MODIS, SEVIRI datasets  (~1 min)
python -m data.generate_icon
python -m data.generate_modis
python -m data.generate_seviri

# 2. Train the conditional diffusion model  (~3–12 min on CPU)
python -m src.train

# 3. Evaluate and generate figures
python -m src.evaluate
```

Outputs:
- `data/synthetic_*.nc`      — synthetic datasets (git-ignored)
- `checkpoints/*.pt`         — model checkpoints (git-ignored)
- `results/figures/*.png`    — comparison and uncertainty figures
- `results/metrics/evaluation.json`

---

## Repository layout

```
modis-10p8um-BT-emulator/
├── configs/
│   └── default.yaml              # hyperparameters, paths, physical-loss weights
├── data/
│   ├── generate_icon.py          # synthetic ICON atmospheric fields
│   ├── generate_modis.py         # MODIS BT via simplified radiative transfer
│   └── generate_seviri.py        # SEVIRI BT via spatial degradation
├── docs/
│   └── methodology.png           # workflow diagram (shown above)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── grid_spec.py              # shared grid specification
│   ├── dataset.py                # PyTorch Dataset (ICON + SEVIRI → MODIS)
│   ├── diffusion_model.py        # conditional DDPM (diffusers)
│   ├── physical_constraints.py   # differentiable physical penalties
│   ├── train.py                  # training loop
│   └── evaluate.py               # metrics + figures
├── results/
│   ├── figures/                  # PNG outputs
│   └── metrics/                  # JSON metrics
├── requirements.txt              # CPU install
├── requirements-gpu.txt          # GPU install override
├── requirements-lock.txt         # exact pinned versions
└── README.md
```

---

## Data description

### Synthetic ICON — `data/synthetic_icon.nc`

Grid: 112 × 138 (2 km), 20 vertical levels up to 20 km, 33 time steps (15 min, 8 h).

| Variable | Shape | Units | Description |
|---|---|---|---|
| `Ts` | (time, lat, lon) | K | Surface temperature |
| `T` | (time, level, lat, lon) | K | Temperature profile |
| `q` | (time, level, lat, lon) | kg/kg | Specific humidity |
| `Lc` | (time, level, lat, lon) | kg/kg | Cloud liquid water |
| `Ic` | (time, level, lat, lon) | kg/kg | Cloud ice |
| `re` | (time, level, lat, lon) | µm | Effective radius |
| `z_top` | (time, lat, lon) | m | Cloud-top height |

### Synthetic MODIS — `data/synthetic_modis.nc`

Grid: 223 × 274 (1 km), one frame per overpass.

| Variable | Shape | Units | Description |
|---|---|---|---|
| `BT` | (time, lat, lon) | K | Band-31 brightness temperature |
| `cloudy` | (time, lat, lon) | 0/1 | Cloud mask |
| `W` | (time, lat, lon) | mm | Column water vapour |
| `Ts_icon` | (time, lat, lon) | K | Surface T for reference |

### Synthetic SEVIRI — `data/synthetic_seviri.nc`

Grid: 75 × 92 (3 km), 33 time steps.

| Variable | Shape | Units | Description |
|---|---|---|---|
| `BT` | (time, lat, lon) | K | 10.8 µm brightness temperature |
| `cloudy` | (time, lat, lon) | 0/1 | Binary cloud mask |
| `cloudy_fraction` | (time, lat, lon) | 0–1 | Fractional coverage |

---

## Model description

**Architecture** — `diffusers.UNet2DConditionModel`, ~5.8 M parameters:

- 4 down / 4 up blocks
- Channel widths `(32, 64, 96, 128)`
- 2 residual layers per block
- Self-attention in the two deepest blocks
- Cross-attention accepting an 8-dim pooled conditioning vector

**Conditioning** — two complementary pathways:

1. Channel concatenation: the noisy target `(B, 1, H, W)` is stacked with the
   8 conditioning channels to form a `(B, 9, H, W)` U-Net input.
2. Cross-attention: the spatial mean of each conditioning channel is passed
   as `encoder_hidden_states` of shape `(B, 1, 8)`.

**Diffusion schedule** — DDPM, 1000 training steps, squared-cosine beta
schedule. Inference uses DDIM with 50 steps.

**Training objective** — epsilon-prediction MSE, optionally augmented with
physical penalties.

---

## Physical constraints

`src/physical_constraints.py` provides four differentiable penalties that
can be added to the training loss:

| Constraint | Purpose | Config key |
|---|---|---|
| **BT bounds** | Penalize values outside `[180, 320] K` | `lambda_bounds` |
| **Spatial smoothness** | Penalize high-frequency noise (Laplacian) | `lambda_smooth` |
| **Temporal smoothness** | Penalize frame-to-frame jumps | `lambda_temporal` |
| **Cloud/BT consistency** | Penalize cold-clear / warm-cloud pixels | `lambda_cloud` |

Hard clipping to `[180, 320] K` is applied at inference.

Disable them for a run with:

```bash
python -m src.train --no-physical
```

---

## Results

After running `python -m src.evaluate`, figures appear under
`results/figures/`:

| Figure | Content |
|---|---|
| `true_vs_generated.png` | Side-by-side true MODIS, ensemble mean, and difference |
| `uncertainty.png` | Ensemble std map + single sample |
| `distribution.png` | Histogram of true vs generated BT |

### Metrics (epoch-10 checkpoint, 500+ CPU training steps)

| Metric | Ensemble mean vs truth | Target |
|---|---|---|
| RMSE | ~10.7 K | 2–6 K |
| Bias | ~−10.4 K | ≈ 0 |
| MAE | ~10.5 K | 1–5 K |
| Wasserstein | ~10.4 K | 1–5 K |
| Correlation | **0.66** | 0.5–0.9 |

**Interpretation.** The model captures spatial patterns well (correlation
0.66) but retains a cold bias from the bimodal brightness-temperature
distribution in the synthetic target. With GPU training, larger models,
and real data, this bias would shrink considerably. See *Limitations* below.

### Loss trajectory

| Training stage | Diffusion loss |
|---|---|
| Untrained | 1.15 |
| 100 steps | 0.24 |
| 300 steps | 0.05 |
| 500 steps | 0.03 |
| 1000 steps | 0.02 |

---

## Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
data:
  tile_size: 64
  batch_size: 4
  samples_per_epoch: 400

model:
  block_out_channels: [32, 64, 96, 128]
  num_train_timesteps: 1000   # DDPM
  num_inference_steps: 50     # DDIM

training:
  n_epochs: 20
  learning_rate: 1.0e-4
  physical:
    enabled: true
    lambda_bounds: 1.0e-3
    lambda_smooth: 1.0e-6
    lambda_cloud:  1.0e-5
```

Command-line overrides take precedence:

```bash
python -m src.train --n-epochs 5 --samples 200 --batch-size 4
python -m src.train --no-physical
```

---

## Known issues

### 1 · Heap corruption on low-core CPUs

Running `src/train.py` for many thousands of iterations can trigger
`corrupted size vs. prev_size` or `corrupted double-linked list` inside
PyTorch's CPU allocator on machines with few cores.

**Fix** — preload Google's tcmalloc:

```bash
sudo apt install -y google-perftools
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python -m src.train
```

Or reduce the glibc arena count:

```bash
MALLOC_ARENA_MAX=1 python -m src.train
```

Not required on GPU hardware.

### 2 · Single-threaded execution

`src/train.py` forces `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` to
avoid the allocator issue. On GPU machines, remove these lines to enable
parallel data loading.

### 3 · Segfault at interpreter exit

`src/evaluate.py` may print `Segmentation fault (core dumped)` **after** all
outputs are written. This is a benign teardown race in PyTorch's C++ layer.

---

## Swapping in real data

The pipeline is designed to run on real ICON/MODIS/SEVIRI data with minimal
changes.

1. **Regrid your inputs** to the same coordinates as `src/grid_spec.py`
   (or modify `GRID` to match your domain).
2. **Match variable names and shapes**:
   - ICON: `Ts`, `T`, `q`, `Lc`, `Ic`, `re`, `z_top` on `(time, [level], lat, lon)`
   - MODIS: `BT` on `(time, lat, lon)`
   - SEVIRI: `BT` on `(time, lat, lon)`
3. **Update paths** in `configs/default.yaml`:
   ```yaml
   paths:
     icon:   /path/to/real_icon.nc
     modis:  /path/to/real_modis.nc
     seviri: /path/to/real_seviri.nc
   ```
4. **Re-tune normalization** in `src/dataset.py` (`NORM_STATS`) to the
   statistics of your real data.
5. **Retrain.** Real data has more variability — expect more epochs and
   possibly a larger model.

No code changes are required in `src/diffusion_model.py`,
`src/physical_constraints.py`, or `src/evaluate.py`.

---

## Limitations

- **Synthetic data.** The ICON/MODIS/SEVIRI fields are statistically inspired
  but not real. Absolute metric values should not be compared with published
  results on real data.
- **Simplified radiative transfer.** Empirical linear clear-sky formula plus
  single-layer cloud approximation. Full RT (RTTOV, CRTM) would produce
  more realistic BT.
- **Limited training.** A few hundred CPU steps. GPU training with more
  data and a larger model would substantially improve accuracy.
- **Cold bias.** Systematic ~−10 K bias due to the bimodal BT distribution
  in the synthetic target. Not a pipeline bug.
- **Single domain.** Central Germany only. The pipeline is
  region-agnostic but has not been tested elsewhere.

---

## Future work

- Train on GPU with `requirements-gpu.txt`.
- Swap synthetic data for real ICON-D2, MODIS `MYD021KM`, and SEVIRI HRIT.
- Add multi-frame (autoregressive) rollout loss for temporal consistency.
- Add differentiable radiative transfer as a physics prior.
- Extend to additional MODIS bands (split-window 11 µm, water-vapour 6.2 µm).
- Compare against GAN / VAE baselines.

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

- [HuggingFace `diffusers`](https://github.com/huggingface/diffusers) — U-Net
  and schedulers.
- [PyTorch](https://pytorch.org/) — autodiff and CPU execution.
- [xarray](https://xarray.dev/) — NetCDF handling.
- DWD ICON, NASA MODIS, EUMETSAT SEVIRI — for the real-world inspiration.
