"""grid_alignment.py — Spatial join antara voxel grid MT dan titik data lain."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from ..utils import seismic_density


def map_points_to_voxel_grid(
    df_mt: pd.DataFrame,
    df_eq: pd.DataFrame,
    radius_meters: float = 100,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
) -> pd.DataFrame:
    """Mengikat titik gempa ke voxel MT menggunakan metode Radius Bola (cKDTree)
    dan menghitung kerapatan seismik kontinu (event/m³) per voxel.

    Parameters
    ----------
    radius_meters : float
        Radius bola pencarian dalam meter. Default 100m.

    Returns
    -------
    pd.DataFrame
        df_mt dengan dua kolom tambahan: `seismic_count` dan `seismic_density`.
    """
    print(
        f"[CONDITIONING] Grid alignment: menghitung gempa dalam radius {radius_meters}m per voxel..."
    )
    df_result = df_mt.copy()

    # Bangun KD-Tree dari koordinat gempa
    coords_eq = df_eq[["X_utm", "Y_utm", "Z_meters"]].values
    tree_eq = cKDTree(coords_eq)

    # Query semua voxel sekaligus
    coords_mt = df_result[[x_col, y_col, z_col]].values
    hits = tree_eq.query_ball_point(coords_mt, r=radius_meters)

    # Hitung count, lalu density (delegasi kalkulasi ke utils)
    n_events = np.array([len(h) for h in hits])
    df_result["seismic_count"] = n_events
    df_result["seismic_density"] = seismic_density(n_events, radius_meters)

    print(
        f"[CONDITIONING] Selesai. Max event dalam satu voxel: {n_events.max()} event."
    )
    return df_result
