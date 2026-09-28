"""
dataset.py
==========

PyTorch Dataset that pairs ICON + SEVIRI conditioning fields with MODIS BT
targets on the MODIS 1 km grid, cropped to 64 x 64 tiles for CPU training.

Pipeline
--------
1. Load synthetic_icon.nc, synthetic_modis.nc, synthetic_seviri.nc.
2. Regrid ICON and SEVIRI fields to the MODIS grid (1 km, 223 x 274).
3. Extract derived ICON channels (surface-layer values, column max of Lc/Ic).
4. Stack channels into a conditioning tensor.
5. Return random 64 x 64 crops.

Conditioning channels (in order)
--------------------------------
  0: Ts             surface temperature            [K]
  1: T_surf         lowest model level T           [K]
  2: q_surf         lowest model level q           [kg/kg]
  3: Lc_colmax      column-max Lc                  [kg/kg]
  4: Ic_colmax      column-max Ic                  [kg/kg]
  5: re_colmax      column-max effective radius    [um]
  6: z_top          cloud-top height               [m]
  7: seviri_bt      SEVIRI 10.8 um BT at time t    [K]

All channels are normalized to roughly [-1, 1] using running statistics
computed once from the data.

Usage
-----
    from src.dataset import ModisIconDataset, build_dataloaders

    train_dl, val_dl = build_dataloaders(batch_size=8)
    for batch in train_dl:
        cond = batch["cond"]         # (B, 8, 64, 64)
        target = batch["target"]     # (B, 1, 64, 64)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, random_split

from src.grid_spec import GRID


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ICON_PATH = Path("data/synthetic_icon.nc")
MODIS_PATH = Path("data/synthetic_modis.nc")
SEVIRI_PATH = Path("data/synthetic_seviri.nc")

TILE_SIZE: int = 64
COND_CHANNELS: int = 8

# Approximate normalization constants (mean, std) per channel.
# Computed once from the synthetic dataset; fixed at construction so
# train and eval agree.
NORM_STATS = {
    "Ts":         (288.0, 5.0),
    "T_surf":     (288.0, 5.0),
    "q_surf":     (0.005, 0.003),
    "Lc_colmax":  (2.0e-4, 3.0e-4),
    "Ic_colmax":  (1.0e-4, 2.0e-4),
    "re_colmax":  (10.0, 5.0),
    "z_top":      (4000.0, 3000.0),
    "seviri_bt":  (270.0, 15.0),
    "modis_bt":   (270.0, 15.0),
}


# ---------------------------------------------------------------------------
# Regridding helpers
# ---------------------------------------------------------------------------

def _regrid_icon_to_modis(
    field_3d: np.ndarray,
    icon_lat: np.ndarray,
    icon_lon: np.ndarray,
    modis_lat: np.ndarray,
    modis_lon: np.ndarray,
) -> np.ndarray:
    """
    Bilinear regridding of an ICON field (time, lat, lon) to the MODIS grid.

    Uses xarray's interp, which handles boundary cases.
    """
    da = xr.DataArray(
        field_3d,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(field_3d.shape[0]),
                "lat": icon_lat, "lon": icon_lon},
    )
    da_modis = da.interp(
        lat=modis_lat, lon=modis_lon, method="linear",
    )
    return da_modis.values.astype(np.float32)


def _regrid_seviri_to_modis(
    field_3d: np.ndarray,
    seviri_lat: np.ndarray,
    seviri_lon: np.ndarray,
    modis_lat: np.ndarray,
    modis_lon: np.ndarray,
) -> np.ndarray:
    """Bilinear regridding of a SEVIRI field (time, lat, lon) to MODIS grid."""
    da = xr.DataArray(
        field_3d,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(field_3d.shape[0]),
                "lat": seviri_lat, "lon": seviri_lon},
    )
    da_modis = da.interp(
        lat=modis_lat, lon=modis_lon, method="linear",
    )
    return da_modis.values.astype(np.float32)


def _normalize(field: np.ndarray, key: str) -> np.ndarray:
    mean, std = NORM_STATS[key]
    return ((field - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ModisIconDataset(Dataset):
    """
    Pairs:
        cond   : (8, H, W)   conditioning tensor (ICON + SEVIRI)
        target : (1, H, W)   MODIS BT (normalized)

    Returns random 64x64 crops from the MODIS grid.
    """

    def __init__(
        self,
        icon_path: Path = ICON_PATH,
        modis_path: Path = MODIS_PATH,
        seviri_path: Path = SEVIRI_PATH,
        tile_size: int = TILE_SIZE,
        samples_per_epoch: int = 400,
        seed: int = 0,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

        # ---- Load datasets ----
        print(f"Loading {icon_path} ...")
        icon = xr.open_dataset(icon_path)
        print(f"Loading {modis_path} ...")
        modis = xr.open_dataset(modis_path)
        print(f"Loading {seviri_path} ...")
        seviri = xr.open_dataset(seviri_path)

        icon_lat = icon["lat"].values
        icon_lon = icon["lon"].values
        seviri_lat = seviri["lat"].values
        seviri_lon = seviri["lon"].values
        modis_lat = modis["lat"].values
        modis_lon = modis["lon"].values

        # ---- MODIS targets ----
        # MODIS has only 2 overpass times. Map them to ICON/SEVIRI time indices
        # 0 and 24 (0.0 h and 6.0 h).
        modis_times = modis["time"].values
        icon_times = icon["time"].values
        modis_icon_idx = [int(np.argmin(np.abs(icon_times - t))) for t in modis_times]
        print(f"MODIS times {list(modis_times)} -> ICON indices {modis_icon_idx}")

        # ---- ICON derived channels ----
        # Shape of icon T etc.: (time, level, lat, lon)
        Ts_icon = icon["Ts"].values                          # (time, lat, lon)
        T_icon = icon["T"].values                            # (time, level, lat, lon)
        q_icon = icon["q"].values
        Lc_icon = icon["Lc"].values
        Ic_icon = icon["Ic"].values
        re_icon = icon["re"].values
        ztop_icon = icon["z_top"].values                     # (time, lat, lon)

        # Surface-layer values (level 0)
        T_surf_icon = T_icon[:, 0, :, :]                     # (time, lat, lon)
        q_surf_icon = q_icon[:, 0, :, :]

        # Column max
        Lc_colmax_icon = Lc_icon.max(axis=1)                 # (time, lat, lon)
        Ic_colmax_icon = Ic_icon.max(axis=1)
        re_colmax_icon = re_icon.max(axis=1)

        # ---- Regrid each ICON channel to MODIS grid ----
        print("Regridding ICON -> MODIS ...")
        Ts_modis = _regrid_icon_to_modis(Ts_icon, icon_lat, icon_lon, modis_lat, modis_lon)
        T_surf_modis = _regrid_icon_to_modis(T_surf_icon, icon_lat, icon_lon, modis_lat, modis_lon)
        q_surf_modis = _regrid_icon_to_modis(q_surf_icon, icon_lat, icon_lon, modis_lat, modis_lon)
        Lc_colmax_modis = _regrid_icon_to_modis(Lc_colmax_icon, icon_lat, icon_lon, modis_lat, modis_lon)
        Ic_colmax_modis = _regrid_icon_to_modis(Ic_colmax_icon, icon_lat, icon_lon, modis_lat, modis_lon)
        re_colmax_modis = _regrid_icon_to_modis(re_colmax_icon, icon_lat, icon_lon, modis_lat, modis_lon)
        ztop_modis = _regrid_icon_to_modis(ztop_icon, icon_lat, icon_lon, modis_lat, modis_lon)

        # ---- Regrid SEVIRI to MODIS grid ----
        print("Regridding SEVIRI -> MODIS ...")
        seviri_bt_modis = _regrid_seviri_to_modis(
            seviri["BT"].values, seviri_lat, seviri_lon, modis_lat, modis_lon,
        )

        # ---- Build tensors: keep only MODIS times ----
        # For each MODIS overpass we take the matching ICON time index and
        # the matching SEVIRI time index (they use the same time grid).
        def take(field, idx_list):
            return field[idx_list]                           # (n_pass, lat, lon)

        # Note: field shape is (time, lat, lon) on MODIS grid
        self.Ts = take(Ts_modis, modis_icon_idx)
        self.T_surf = take(T_surf_modis, modis_icon_idx)
        self.q_surf = take(q_surf_modis, modis_icon_idx)
        self.Lc_colmax = take(Lc_colmax_modis, modis_icon_idx)
        self.Ic_colmax = take(Ic_colmax_modis, modis_icon_idx)
        self.re_colmax = take(re_colmax_modis, modis_icon_idx)
        self.z_top = take(ztop_modis, modis_icon_idx)
        self.seviri_bt = take(seviri_bt_modis, modis_icon_idx)

        # Normalize each channel
        self.Ts = _normalize(self.Ts, "Ts")
        self.T_surf = _normalize(self.T_surf, "T_surf")
        self.q_surf = _normalize(self.q_surf, "q_surf")
        self.Lc_colmax = _normalize(self.Lc_colmax, "Lc_colmax")
        self.Ic_colmax = _normalize(self.Ic_colmax, "Ic_colmax")
        self.re_colmax = _normalize(self.re_colmax, "re_colmax")
        self.z_top = _normalize(self.z_top, "z_top")
        self.seviri_bt = _normalize(self.seviri_bt, "seviri_bt")

        # Stack conditioning: (n_pass, 8, lat, lon)
        self.cond = np.stack(
            [
                self.Ts, self.T_surf, self.q_surf,
                self.Lc_colmax, self.Ic_colmax, self.re_colmax,
                self.z_top, self.seviri_bt,
            ],
            axis=1,
        ).astype(np.float32)

        # Normalize target
        self.target = _normalize(modis["BT"].values, "modis_bt")  # (n_pass, lat, lon)
        self.target = self.target[:, None, :, :]                  # (n_pass, 1, lat, lon)

        # Convert to torch
        self.cond = torch.from_numpy(self.cond)                   # (n_pass, 8, H, W)
        self.target = torch.from_numpy(self.target)               # (n_pass, 1, H, W)

        self.n_pass, _, self.H, self.W = self.cond.shape
        print(f"Loaded: {self.n_pass} pass, cond {self.cond.shape}, target {self.target.shape}")

        # --------------- Sanity: channels must be finite ---------------
        assert torch.isfinite(self.cond).all(), "non-finite values in cond"
        assert torch.isfinite(self.target).all(), "non-finite values in target"

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        # Random pass and tile
        ip = self.rng.integers(0, self.n_pass)
        i = self.rng.integers(0, self.H - self.tile_size + 1)
        j = self.rng.integers(0, self.W - self.tile_size + 1)

        cond = self.cond[ip, :, i:i+self.tile_size, j:j+self.tile_size]
        target = self.target[ip, :, i:i+self.tile_size, j:j+self.tile_size]

        return {
            "cond":   cond.clone(),
            "target": target.clone(),
            "meta":   {"pass": int(ip), "row": int(i), "col": int(j)},
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size: int = 8,
    val_fraction: float = 0.1,
    samples_per_epoch: int = 400,
    tile_size: int = TILE_SIZE,
    seed: int = 0,
    num_workers: int = 0,
):
    """
    Build train and val DataLoaders.

    Because the dataset returns random crops on the fly, we use a single
    dataset and split the number of samples.
    """
    ds = ModisIconDataset(tile_size=tile_size, samples_per_epoch=samples_per_epoch,
                          seed=seed)

    n_total = samples_per_epoch
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building dataset ...")
    ds = ModisIconDataset(samples_per_epoch=8)

    print(f"\nlen(dataset) = {len(ds)}")
    item = ds[0]
    print(f"cond shape   = {tuple(item['cond'].shape)}")
    print(f"target shape = {tuple(item['target'].shape)}")
    print(f"cond  mean / std = {item['cond'].mean():+.3f} / {item['cond'].std():.3f}")
    print(f"target mean / std = {item['target'].mean():+.3f} / {item['target'].std():.3f}")
    print(f"cond finite : {torch.isfinite(item['cond']).all().item()}")
    print(f"target finite: {torch.isfinite(item['target']).all().item()}")

    print("\nBuilding dataloaders ...")
    train_dl, val_dl = build_dataloaders(batch_size=4, samples_per_epoch=32)
    batch = next(iter(train_dl))
    print(f"batch cond shape   = {tuple(batch['cond'].shape)}")
    print(f"batch target shape = {tuple(batch['target'].shape)}")
    print("\nAll dataset sanity checks passed.")
