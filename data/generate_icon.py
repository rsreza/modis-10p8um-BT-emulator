"""
generate_icon.py
================

Generate synthetic ICON-D2-like atmospheric fields on the project grid.

Produces a NetCDF file:  data/synthetic_icon.nc

Dimensions:  time (33), level (20), lat (112), lon (138)

Variables:
    Ts     (time, lat, lon)        surface temperature               [K]
    T      (time, level, lat, lon) temperature profile                [K]
    q      (time, level, lat, lon) specific humidity                  [kg/kg]
    Lc     (time, level, lat, lon) cloud liquid water content         [kg/kg]
    Ic     (time, level, lat, lon) cloud ice content                  [kg/kg]
    re     (time, level, lat, lon) cloud effective radius             [um]
    z_top  (time, lat, lon)        cloud-top height                   [m]

Usage:
    python -m data.generate_icon
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from src.grid_spec import GRID


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED: int = 42

# Physics constants
T_SURFACE_MEAN_K: float = 288.0
DIURNAL_AMPLITUDE_K: float = 6.0
LAPSE_RATE_K_PER_KM: float = 6.5
TROPOPAUSE_KM: float = 11.0

HUMIDITY_SCALE_HEIGHT_M: float = 2000.0
Q_SURFACE_MEAN: float = 0.005

INVERSION_STRENGTH_K: float = 3.0
INVERSION_HEIGHT_M: float = 300.0

WIND_U_MS: float = 10.0
WIND_V_MS: float = 2.0

# --- Cloud statistics (tuned for realistic coverage) ---
N_CLOUD_BLOBS: int = 12
CLOUD_PEAK_HEIGHT_RANGE_M: tuple = (2000.0, 5000.0)
ICE_PEAK_HEIGHT_RANGE_M: tuple = (5000.0, 9000.0)
CLOUD_HORIZONTAL_SIGMA_KM: tuple = (10.0, 30.0)
CLOUD_VERTICAL_SIGMA_KM: float = 0.6
LWC_PEAK_KG_PER_KG: float = 5.0e-4
IWC_PEAK_KG_PER_KG: float = 2.0e-4

CLOUD_THRESHOLD_KG_PER_KG: float = 3.0e-4    # 0.3 g/kg

EFFECTIVE_RADIUS_MIN_UM: float = 5.0
EFFECTIVE_RADIUS_MAX_UM: float = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _km_per_deg_latlon(lat_mean_deg: float) -> tuple[float, float]:
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(lat_mean_deg))
    return km_per_deg_lat, km_per_deg_lon


def _gaussian_random_field(
    shape: tuple[int, int],
    sigma_grid: float,
    rng: np.random.Generator,
) -> np.ndarray:
    white = rng.standard_normal(shape)
    return gaussian_filter(white, sigma=sigma_grid, mode="wrap")


# ---------------------------------------------------------------------------
# Vertical structure
# ---------------------------------------------------------------------------

def _temperature_profile(
    z_m: np.ndarray,
    t_surface_k: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    z_km = z_m / 1000.0
    n_levels = z_m.size
    n_lat, n_lon = t_surface_k.shape

    T = np.empty((n_levels, n_lat, n_lon), dtype=np.float32)
    for k, zk in enumerate(z_km):
        if zk <= TROPOPAUSE_KM:
            T[k] = t_surface_k - LAPSE_RATE_K_PER_KM * zk
        else:
            T[k] = t_surface_k - LAPSE_RATE_K_PER_KM * TROPOPAUSE_KM

    for k in range(n_levels):
        noise = _gaussian_random_field((n_lat, n_lon), sigma_grid=3.0, rng=rng)
        T[k] += 1.5 * noise

    inversion_mask = z_m <= INVERSION_HEIGHT_M
    if inversion_mask.any():
        inversion_profile = INVERSION_STRENGTH_K * (
            1.0 - z_m[inversion_mask] / INVERSION_HEIGHT_M
        )
        T[inversion_mask] += inversion_profile[:, None, None]

    return T


def _humidity_profile(
    z_m: np.ndarray,
    temperature: np.ndarray,
    cloud_mask_2d: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    z_km = z_m / 1000.0
    scale_height_km = HUMIDITY_SCALE_HEIGHT_M / 1000.0
    n_levels, n_lat, n_lon = temperature.shape

    q = Q_SURFACE_MEAN * np.exp(-z_km / scale_height_km)
    q = np.broadcast_to(q[:, None, None], (n_levels, n_lat, n_lon)).copy()

    for k in range(n_levels):
        noise = _gaussian_random_field((n_lat, n_lon), sigma_grid=8.0, rng=rng)
        q[k] *= (1.0 + 0.5 * noise)
        q[k] = np.clip(q[k], 1.0e-6, None)

    q *= (1.0 + 0.3 * cloud_mask_2d[None, :, :])

    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# Cloud generation
# ---------------------------------------------------------------------------

def _place_blob(
    center_lat_idx: float,
    center_lon_idx: float,
    sigma_lat_idx: float,
    sigma_lon_idx: float,
    shape: tuple[int, int],
) -> np.ndarray:
    n_lat, n_lon = shape
    yy, xx = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    dx = np.minimum(
        np.abs(xx - center_lon_idx),
        n_lon - np.abs(xx - center_lon_idx),
    )
    dy = yy - center_lat_idx
    return np.exp(-0.5 * ((dx / sigma_lon_idx) ** 2 + (dy / sigma_lat_idx) ** 2))


def _vertical_gaussian(z_m: np.ndarray, peak_m: float, sigma_m: float) -> np.ndarray:
    return np.exp(-0.5 * ((z_m - peak_m) / sigma_m) ** 2)


def _generate_clouds_at_time(
    time_index: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_levels = GRID.n_levels
    n_lat = len(GRID.lat_icon)
    n_lon = len(GRID.lon_icon)
    z_m = GRID.level_heights_m

    lat_mean = 0.5 * (GRID.lat_min + GRID.lat_max)
    km_per_deg_lat, km_per_deg_lon = _km_per_deg_latlon(lat_mean)

    dy_km = (GRID.lat_icon[1] - GRID.lat_icon[0]) * km_per_deg_lat
    dx_km = (GRID.lon_icon[1] - GRID.lon_icon[0]) * km_per_deg_lon

    t_seconds = time_index * GRID.dt_minutes * 60.0
    advect_x_km = WIND_U_MS * t_seconds / 1000.0
    advect_y_km = WIND_V_MS * t_seconds / 1000.0

    Lc = np.zeros((n_levels, n_lat, n_lon), dtype=np.float32)
    Ic = np.zeros((n_levels, n_lat, n_lon), dtype=np.float32)

    for _ in range(N_CLOUD_BLOBS):
        c_lat_idx = rng.uniform(0, n_lat - 1)
        c_lon_idx = rng.uniform(0, n_lon - 1)

        c_lat_idx -= advect_y_km / dy_km
        c_lon_idx += advect_x_km / dx_km

        sigma_km = rng.uniform(*CLOUD_HORIZONTAL_SIGMA_KM)
        sigma_lat_idx = sigma_km / dy_km
        sigma_lon_idx = sigma_km / dx_km

        blob2d = _place_blob(
            c_lat_idx, c_lon_idx, sigma_lat_idx, sigma_lon_idx, (n_lat, n_lon)
        )

        lc_peak_m = rng.uniform(*CLOUD_PEAK_HEIGHT_RANGE_M)
        vert_lc = _vertical_gaussian(z_m, lc_peak_m, CLOUD_VERTICAL_SIGMA_KM * 1000.0)

        ic_peak_m = rng.uniform(*ICE_PEAK_HEIGHT_RANGE_M)
        vert_ic = _vertical_gaussian(z_m, ic_peak_m, CLOUD_VERTICAL_SIGMA_KM * 1000.0)

        lc_amplitude = LWC_PEAK_KG_PER_KG * rng.uniform(0.5, 1.5)
        ic_amplitude = IWC_PEAK_KG_PER_KG * rng.uniform(0.5, 1.5)

        Lc += lc_amplitude * vert_lc[:, None, None] * blob2d[None, :, :]
        Ic += ic_amplitude * vert_ic[:, None, None] * blob2d[None, :, :]

    for k in range(n_levels):
        Lc[k] *= (1.0 + 0.1 * _gaussian_random_field((n_lat, n_lon), 1.5, rng))
        Ic[k] *= (1.0 + 0.1 * _gaussian_random_field((n_lat, n_lon), 1.5, rng))

    Lc = np.clip(Lc, 0.0, None).astype(np.float32)
    Ic = np.clip(Ic, 0.0, None).astype(np.float32)

    lc_norm = Lc / (LWC_PEAK_KG_PER_KG + 1e-12)
    re = EFFECTIVE_RADIUS_MIN_UM + (
        EFFECTIVE_RADIUS_MAX_UM - EFFECTIVE_RADIUS_MIN_UM
    ) * np.clip(lc_norm, 0.0, 1.0)

    return Lc, Ic, re.astype(np.float32)


# ---------------------------------------------------------------------------
# Surface temperature
# ---------------------------------------------------------------------------

def _surface_temperature(
    time_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_lat = len(GRID.lat_icon)
    n_lon = len(GRID.lon_icon)

    t_hours = time_index * GRID.dt_minutes / 60.0
    phase = np.cos((t_hours - 15.0) / 24.0 * 2.0 * np.pi)
    diurnal = DIURNAL_AMPLITUDE_K * phase

    background = T_SURFACE_MEAN_K + 1.0 * _gaussian_random_field(
        (n_lat, n_lon), sigma_grid=5.0, rng=rng
    )

    return (background + diurnal).astype(np.float32)


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

def generate_icon_dataset(seed: int = SEED) -> xr.Dataset:
    rng = np.random.default_rng(seed)

    n_times = GRID.n_times
    n_levels = GRID.n_levels
    n_lat = len(GRID.lat_icon)
    n_lon = len(GRID.lon_icon)

    Ts = np.empty((n_times, n_lat, n_lon), dtype=np.float32)
    T = np.empty((n_times, n_levels, n_lat, n_lon), dtype=np.float32)
    q = np.empty((n_times, n_levels, n_lat, n_lon), dtype=np.float32)
    Lc = np.empty((n_times, n_levels, n_lat, n_lon), dtype=np.float32)
    Ic = np.empty((n_times, n_levels, n_lat, n_lon), dtype=np.float32)
    re = np.empty((n_times, n_levels, n_lat, n_lon), dtype=np.float32)
    z_top = np.empty((n_times, n_lat, n_lon), dtype=np.float32)

    for it in range(n_times):
        print(f"  generating t={it+1:02d}/{n_times}  "
              f"({it * GRID.dt_minutes / 60.0:.2f} h)")
        Ts[it] = _surface_temperature(it, rng)
        T[it] = _temperature_profile(GRID.level_heights_m, Ts[it], rng)

        Lc_it, Ic_it, re_it = _generate_clouds_at_time(it, rng)
        Lc[it] = Lc_it
        Ic[it] = Ic_it
        re[it] = re_it

        cloud_mask_2d = (Lc_it + Ic_it > CLOUD_THRESHOLD_KG_PER_KG).any(axis=0).astype(np.float32)
        q[it] = _humidity_profile(GRID.level_heights_m, T[it], cloud_mask_2d, rng)

        cloudy = (Lc_it + Ic_it) > CLOUD_THRESHOLD_KG_PER_KG
        z_top_it = np.zeros((n_lat, n_lon), dtype=np.float32)
        for k in range(n_levels - 1, -1, -1):
            newly = cloudy[k] & (z_top_it == 0.0)
            z_top_it[newly] = GRID.level_heights_m[k]
        z_top[it] = z_top_it

    ds = xr.Dataset(
        data_vars={
            "Ts":    (("time", "lat", "lon"), Ts),
            "T":     (("time", "level", "lat", "lon"), T),
            "q":     (("time", "level", "lat", "lon"), q),
            "Lc":    (("time", "level", "lat", "lon"), Lc),
            "Ic":    (("time", "level", "lat", "lon"), Ic),
            "re":    (("time", "level", "lat", "lon"), re),
            "z_top": (("time", "lat", "lon"), z_top),
        },
        coords=GRID.icon_coords(),
        attrs={
            "description": "Synthetic ICON-D2-like atmospheric fields",
            "domain": f"lat [{GRID.lat_min}, {GRID.lat_max}], "
                      f"lon [{GRID.lon_min}, {GRID.lon_max}]",
            "seed": seed,
            "generator": "data/generate_icon.py",
        },
    )
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ICON fields")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/synthetic_icon.nc"),
        help="output NetCDF path",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic ICON dataset  (seed={args.seed})")
    print(f"  grid : {len(GRID.lat_icon)} x {len(GRID.lon_icon)}, "
          f"{GRID.n_levels} levels, {GRID.n_times} times")
    ds = generate_icon_dataset(seed=args.seed)

    print(f"\nWriting {args.out} ...")
    ds.to_netcdf(args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Done.  File size: {size_mb:.1f} MB")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Shapes   : time={GRID.n_times}, level={GRID.n_levels}, "
          f"lat={len(GRID.lat_icon)}, lon={len(GRID.lon_icon)}")


if __name__ == "__main__":
    main()
