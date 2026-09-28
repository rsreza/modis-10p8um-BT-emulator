"""
Grid specification for the diffusion-based ICON->MODIS emulator.

Defines the common spatial and temporal grid shared by:
  - synthetic ICON fields
  - synthetic MODIS brightness temperature
  - synthetic SEVIRI brightness temperature

All other modules import from here so the domain is defined in exactly one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# 1. Horizontal domain
# ---------------------------------------------------------------------------

LAT_MIN: float = 51.0
LAT_MAX: float = 53.0
LON_MIN: float = 8.0
LON_MAX: float = 12.0

ICON_RES_KM: float = 2.0
MODIS_RES_KM: float = 1.0
SEVIRI_RES_KM: float = 3.0

KM_PER_DEG_LAT: float = 111.0
KM_PER_DEG_LON: float = 111.0 * math.cos(math.radians(0.5 * (LAT_MIN + LAT_MAX)))


# ---------------------------------------------------------------------------
# 2. Vertical grid
# ---------------------------------------------------------------------------

N_LEVELS: int = 20
Z_TOP_M: float = 20_000.0

LEVEL_HEIGHTS_M: np.ndarray = np.linspace(0.0, Z_TOP_M, N_LEVELS)


# ---------------------------------------------------------------------------
# 3. Temporal grid
# ---------------------------------------------------------------------------

FORECAST_HOURS: int = 8
DT_MINUTES: int = 15
N_TIMES: int = FORECAST_HOURS * 60 // DT_MINUTES + 1  # 33

MODIS_OVERPASS_HOURS: Tuple[int, ...] = (0, 6, 12, 18)


# ---------------------------------------------------------------------------
# 4. Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridSpec:
    """Immutable container describing the grid."""

    lat_min: float = LAT_MIN
    lat_max: float = LAT_MAX
    lon_min: float = LON_MIN
    lon_max: float = LON_MAX

    icon_res_km: float = ICON_RES_KM
    modis_res_km: float = MODIS_RES_KM
    seviri_res_km: float = SEVIRI_RES_KM

    n_levels: int = N_LEVELS
    z_top_m: float = Z_TOP_M
    level_heights_m: np.ndarray = field(default_factory=lambda: LEVEL_HEIGHTS_M.copy())

    n_times: int = N_TIMES
    dt_minutes: int = DT_MINUTES
    modis_overpass_hours: Tuple[int, ...] = MODIS_OVERPASS_HOURS

    # ---------------- derived quantities ----------------

    @property
    def lat_icon(self) -> np.ndarray:
        n = self._n_points(self.lat_max - self.lat_min, self.icon_res_km, axis="lat")
        return np.linspace(self.lat_min, self.lat_max, n)

    @property
    def lon_icon(self) -> np.ndarray:
        n = self._n_points(self.lon_max - self.lon_min, self.icon_res_km, axis="lon")
        return np.linspace(self.lon_min, self.lon_max, n)

    @property
    def lat_modis(self) -> np.ndarray:
        n = self._n_points(self.lat_max - self.lat_min, self.modis_res_km, axis="lat")
        return np.linspace(self.lat_min, self.lat_max, n)

    @property
    def lon_modis(self) -> np.ndarray:
        n = self._n_points(self.lon_max - self.lon_min, self.modis_res_km, axis="lon")
        return np.linspace(self.lon_min, self.lon_max, n)

    @property
    def lat_seviri(self) -> np.ndarray:
        n = self._n_points(self.lat_max - self.lat_min, self.seviri_res_km, axis="lat")
        return np.linspace(self.lat_min, self.lat_max, n)

    @property
    def lon_seviri(self) -> np.ndarray:
        n = self._n_points(self.lon_max - self.lon_min, self.seviri_res_km, axis="lon")
        return np.linspace(self.lon_min, self.lon_max, n)

    @property
    def times_hours(self) -> np.ndarray:
        return np.arange(self.n_times) * self.dt_minutes / 60.0

    # ---------------- helpers ----------------

    @staticmethod
    def _n_points(span_deg: float, res_km: float, axis: str) -> int:
        km_per_deg = KM_PER_DEG_LAT if axis == "lat" else KM_PER_DEG_LON
        span_km = span_deg * km_per_deg
        return int(round(span_km / res_km)) + 1

    # ---------------- xarray helpers ----------------

    def icon_coords(self) -> dict:
        return {
            "time": self.times_hours,
            "level": self.level_heights_m,
            "lat": self.lat_icon,
            "lon": self.lon_icon,
        }

    def modis_coords(self) -> dict:
        return {
            "time": self.times_hours,
            "lat": self.lat_modis,
            "lon": self.lon_modis,
        }

    def seviri_coords(self) -> dict:
        return {
            "time": self.times_hours,
            "lat": self.lat_seviri,
            "lon": self.lon_seviri,
        }

    def describe(self) -> str:
        return (
            f"GridSpec\n"
            f"  domain      : lat [{self.lat_min}, {self.lat_max}], "
            f"lon [{self.lon_min}, {self.lon_max}]\n"
            f"  ICON grid   : {len(self.lat_icon)} x {len(self.lon_icon)} "
            f"({self.icon_res_km} km)\n"
            f"  MODIS grid  : {len(self.lat_modis)} x {len(self.lon_modis)} "
            f"({self.modis_res_km} km)\n"
            f"  SEVIRI grid : {len(self.lat_seviri)} x {len(self.lon_seviri)} "
            f"({self.seviri_res_km} km)\n"
            f"  levels      : {self.n_levels} (0-{self.z_top_m/1000:.0f} km)\n"
            f"  time steps  : {self.n_times} ({self.dt_minutes}-min, "
            f"{FORECAST_HOURS} h)\n"
            f"  MODIS passes: {self.modis_overpass_hours} UTC"
        )


# ---------------------------------------------------------------------------
# 5. Default instance
# ---------------------------------------------------------------------------

GRID = GridSpec()


# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(GRID.describe())

    assert len(GRID.lat_icon) > 100, "ICON lat grid too small"
    assert len(GRID.lon_icon) > 100, "ICON lon grid too small"
    assert GRID.n_levels == 20
    assert GRID.n_times == 33
    assert 0.0 in GRID.level_heights_m
    assert GRID.z_top_m in GRID.level_heights_m
    print("\nAll grid sanity checks passed.")
