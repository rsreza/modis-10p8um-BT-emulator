"""
generate_modis.py
=================

Generate synthetic MODIS Band-31 (10.8 um) brightness temperature from the
synthetic ICON fields produced by ``data/generate_icon.py``.

Approach
--------
Clear sky:   BT_clear  = Ts - Gamma * W + clear_sky_offset
Cloudy:      BT_cloudy = T(z_top) + epsilon

Cloud mask is computed on the ICON grid, upsampled with NEAREST-NEIGHBOUR
interpolation to MODIS (preserves total cloud fraction), and stored as
uint8 {0, 1}.

Output
------
data/synthetic_modis.nc with variables BT, cloudy, Ts_icon, W.

Usage:
    python -m data.generate_modis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from src.grid_spec import GRID


# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

GAMMA_K_PER_MM: float = 0.3
CLEAR_SKY_OFFSET_K: float = 3.0
SENSOR_NOISE_K: float = 0.2
SPATIAL_NOISE_K: float = 0.5
CLOUD_EPSILON_K: float = 0.5

# Must match generate_icon.py
CLOUD_THRESHOLD_KG_PER_KG: float = 3.0e-4

BT_MIN_K: float = 180.0
BT_MAX_K: float = 320.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bilinear_to_modis(field: np.ndarray) -> np.ndarray:
    """Bilinear interpolation from ICON to MODIS grid (continuous fields)."""
    da = xr.DataArray(
        field,
        dims=("lat", "lon"),
        coords={"lat": GRID.lat_icon, "lon": GRID.lon_icon},
    )
    da_modis = da.interp(
        lat=GRID.lat_modis,
        lon=GRID.lon_modis,
        method="linear",
    )
    return da_modis.values


def _nearest_to_modis(field: np.ndarray) -> np.ndarray:
    """Nearest-neighbour interpolation from ICON to MODIS grid (categorical)."""
    da = xr.DataArray(
        field,
        dims=("lat", "lon"),
        coords={"lat": GRID.lat_icon, "lon": GRID.lon_icon},
    )
    da_modis = da.interp(
        lat=GRID.lat_modis,
        lon=GRID.lon_modis,
        method="nearest",
    )
    return da_modis.values


def _column_water_vapour_mm(q: np.ndarray) -> np.ndarray:
    """Integrate q over height -> column water vapour [kg/m^2 == mm]."""
    z_m = GRID.level_heights_m
    rho0 = 1.2
    H = 8000.0
    rho = rho0 * np.exp(-z_m / H)

    if len(z_m) > 1:
        dz = np.gradient(z_m)
    else:
        dz = np.array([1.0])

    W = np.tensordot(rho * dz, q, axes=([0], [0]))
    return W.astype(np.float32)


def _cloud_mask_and_top(
    Lc: np.ndarray, Ic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D bool cloud mask and 2-D int cloud-top level index."""
    n_levels, n_lat, n_lon = Lc.shape
    cloudy_3d = (Lc + Ic) > CLOUD_THRESHOLD_KG_PER_KG
    cloudy = cloudy_3d.any(axis=0)

    z_top_idx = np.full((n_lat, n_lon), -1, dtype=np.int16)
    for k in range(n_levels - 1, -1, -1):
        newly = cloudy_3d[k] & (z_top_idx == -1)
        z_top_idx[newly] = k

    return cloudy, z_top_idx


def _interp_temperature_at_level(
    T: np.ndarray, level_idx: np.ndarray
) -> np.ndarray:
    """T[level_idx[lat,lon], lat, lon] with safe clipping."""
    n_levels, n_lat, n_lon = T.shape
    safe_idx = np.clip(level_idx, 0, n_levels - 1)
    yy, xx = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    return T[safe_idx, yy, xx]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_modis_dataset(
    icon_path: Path,
    overpass_hours: tuple[int, ...] = (0, 6, 12, 18),
) -> xr.Dataset:
    print(f"Reading {icon_path} ...")
    icon = xr.open_dataset(icon_path)

    times_h = icon.time.values
    overpass_idx = []
    for h in overpass_hours:
        i = int(np.argmin(np.abs(times_h - h)))
        if abs(times_h[i] - h) <= 1e-6:
            overpass_idx.append(i)
    overpass_idx = sorted(set(overpass_idx))

    if not overpass_idx:
        raise ValueError("No overpass times fall inside the generated window.")

    print(f"MODIS overpasses at hours "
          f"{[float(times_h[i]) for i in overpass_idx]} "
          f"-> indices {overpass_idx}")

    n_pass = len(overpass_idx)
    n_lat_m = len(GRID.lat_modis)
    n_lon_m = len(GRID.lon_modis)

    BT = np.zeros((n_pass, n_lat_m, n_lon_m), dtype=np.float32)
    CLD = np.zeros((n_pass, n_lat_m, n_lon_m), dtype=np.uint8)
    TS_out = np.zeros((n_pass, n_lat_m, n_lon_m), dtype=np.float32)
    W_out = np.zeros((n_pass, n_lat_m, n_lon_m), dtype=np.float32)

    rng = np.random.default_rng(1234)

    for ip, it in enumerate(overpass_idx):
        print(f"  overpass {ip+1}/{n_pass}  (t index {it}, "
              f"{float(times_h[it]):.2f} h)")

        Ts = icon["Ts"].isel(time=it).values
        T = icon["T"].isel(time=it).values
        q = icon["q"].isel(time=it).values
        Lc = icon["Lc"].isel(time=it).values
        Ic = icon["Ic"].isel(time=it).values

        W = _column_water_vapour_mm(q)
        cloudy_icon, z_top_idx = _cloud_mask_and_top(Lc, Ic)
        T_cloud_top = _interp_temperature_at_level(T, z_top_idx)

        BT_clear_icon = Ts - GAMMA_K_PER_MM * W + CLEAR_SKY_OFFSET_K
        BT_cloudy_icon = T_cloud_top + CLOUD_EPSILON_K * rng.standard_normal(
            T_cloud_top.shape
        ).astype(np.float32)

        BT_icon = np.where(cloudy_icon, BT_cloudy_icon, BT_clear_icon).astype(np.float32)
        BT_icon += SPATIAL_NOISE_K * gaussian_filter(
            rng.standard_normal(BT_icon.shape).astype(np.float32),
            sigma=1.0,
        )

        CLD_modis = _nearest_to_modis(cloudy_icon.astype(np.uint8))

        BT_modis = _bilinear_to_modis(BT_icon)
        Ts_modis = _bilinear_to_modis(Ts)
        W_modis = _bilinear_to_modis(W)

        BT_modis += SENSOR_NOISE_K * rng.standard_normal(BT_modis.shape).astype(np.float32)
        BT_modis = np.clip(BT_modis, BT_MIN_K, BT_MAX_K)

        BT[ip] = BT_modis.astype(np.float32)
        CLD[ip] = CLD_modis
        TS_out[ip] = Ts_modis.astype(np.float32)
        W_out[ip] = W_modis.astype(np.float32)

    modis_times = np.array([float(times_h[i]) for i in overpass_idx])

    ds = xr.Dataset(
        data_vars={
            "BT":      (("time", "lat", "lon"), BT),
            "cloudy":  (("time", "lat", "lon"), CLD),
            "Ts_icon": (("time", "lat", "lon"), TS_out),
            "W":       (("time", "lat", "lon"), W_out),
        },
        coords={
            "time": modis_times,
            "lat": GRID.lat_modis,
            "lon": GRID.lon_modis,
        },
        attrs={
            "description": "Synthetic MODIS Band-31 (10.8 um) brightness temperature",
            "source": "generated from synthetic ICON fields via simplified RT",
            "gamma_K_per_mm": GAMMA_K_PER_MM,
            "clear_sky_offset_K": CLEAR_SKY_OFFSET_K,
            "sensor_noise_K": SENSOR_NOISE_K,
            "spatial_noise_K": SPATIAL_NOISE_K,
        },
    )
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MODIS BT")
    parser.add_argument(
        "--in", dest="in_path", type=Path,
        default=Path("data/synthetic_icon.nc"),
        help="input ICON NetCDF",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/synthetic_modis.nc"),
        help="output MODIS NetCDF",
    )
    parser.add_argument(
        "--overpass-hours", type=int, nargs="+",
        default=[0, 6, 12, 18],
        help="UTC hours of MODIS overpasses",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = generate_modis_dataset(
        args.in_path, overpass_hours=tuple(args.overpass_hours)
    )

    print(f"\nWriting {args.out} ...")
    ds.to_netcdf(args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Done.  File size: {size_mb:.1f} MB")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Shapes   : time={ds.sizes['time']}, "
          f"lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")


if __name__ == "__main__":
    main()
