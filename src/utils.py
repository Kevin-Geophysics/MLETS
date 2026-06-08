"""utils.py — Pure utility functions: matematis, spasial, dan konversi universal.

Tidak ada konteks geologi di sini. Semua fungsi bersifat stateless:
menerima angka / array, mengembalikan angka / array.
Mudah di-unit test secara independen.
"""

import numpy as np
from pyproj import Transformer


# ---------------------------------------------------------------------------
# MATEMATIS
# ---------------------------------------------------------------------------

def sphere_volume(radius: float) -> float:
    """Menghitung volume bola dengan radius tertentu (dalam meter³).

    V = (4/3) * π * r³
    """
    return (4 / 3) * np.pi * (radius ** 3)


def seismic_density(n_events: np.ndarray, radius: float) -> np.ndarray:
    """Menghitung kerapatan seismik kontinu (event/m³).

    Parameters
    ----------
    n_events : np.ndarray
        Jumlah event gempa per voxel.
    radius : float
        Radius bola pencarian dalam meter.

    Returns
    -------
    np.ndarray
        Nilai kerapatan seismik per voxel (event/m³).
    """
    vol = sphere_volume(radius)
    return n_events / vol


def log10_safe(values: np.ndarray) -> np.ndarray:
    """Log10 yang aman: mengembalikan NaN untuk nilai <= 0.

    Menggantikan filter baris (drop row) agar shape array tetap terjaga,
    keputusan drop/keep diserahkan ke layer conditioning.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(values > 0, np.log10(values), np.nan)
    return result


# ---------------------------------------------------------------------------
# KONVERSI KOORDINAT
# ---------------------------------------------------------------------------

def latlon_to_utm(
    lon: np.ndarray,
    lat: np.ndarray,
    utm_zone: int,
    hemisphere: str = "N",
) -> tuple[np.ndarray, np.ndarray]:
    """Mengonversi koordinat Lat/Lon (WGS84) ke UTM meter.

    Parameters
    ----------
    lon, lat : np.ndarray
        Koordinat dalam derajat desimal (WGS84).
    utm_zone : int
        Nomor zona UTM (1–60).
    hemisphere : str
        'N' untuk Northern Hemisphere, 'S' untuk Southern Hemisphere.

    Returns
    -------
    x_utm, y_utm : tuple of np.ndarray
        Koordinat dalam satuan meter.
    """
    hemi_code = "6" if hemisphere.upper() == "N" else "7"
    epsg_utm = f"epsg:32{hemi_code}{utm_zone:02d}"
    transformer = Transformer.from_crs("epsg:4326", epsg_utm, always_xy=True)
    x_utm, y_utm = transformer.transform(lon, lat)
    return np.array(x_utm), np.array(y_utm)


def depth_km_to_z_meters(depth_km: np.ndarray) -> np.ndarray:
    """Mengonversi kedalaman USGS (km, positif ke bawah) ke konvensi Z grid MT
    (meter, negatif ke bawah).

    depth_km=1.5  →  Z_meters=-1500
    """
    return -1.0 * (depth_km * 1000)
