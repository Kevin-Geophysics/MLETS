"""well_validation.py — Validasi silang hasil klaster HDBSCAN dengan data log sumur bor."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def extract_cluster_along_well(
    df_clustered: pd.DataFrame,
    df_well: pd.DataFrame,
    well_xyz_cols: list = ["X", "Y", "Z"],
    grid_xyz_cols: list = ["X", "Y", "Z"],
    max_distance: float = 100,
) -> pd.DataFrame:
    """Memetakan label klaster HDBSCAN ke sepanjang jalur sumur bor
    menggunakan Nearest Neighbor (cKDTree).

    Parameters
    ----------
    max_distance : float
        Toleransi jarak maksimal (meter) antara titik sumur dan voxel MT terdekat.
        Titik sumur yang melebihi batas ini dianggap out-of-bounds dan di-drop.
    """
    print("[VALIDATION] Memetakan klaster HDBSCAN ke jalur sumur bor...")

    coords_grid = df_clustered[grid_xyz_cols].values
    tree_grid = cKDTree(coords_grid)

    coords_well = df_well[well_xyz_cols].values
    distances, indices = tree_grid.query(coords_well, k=1)

    # Ambil label langsung dari numpy array — aman terhadap non-contiguous index DataFrame
    cluster_labels_array = df_clustered["cluster_labels"].values

    df_result = df_well.copy()
    df_result["assigned_cluster"] = cluster_labels_array[indices]
    df_result["distance_to_voxel"] = distances

    # Drop titik yang out-of-bounds
    n_before = len(df_result)
    df_result = df_result[df_result["distance_to_voxel"] <= max_distance].copy()
    n_dropped = n_before - len(df_result)
    if n_dropped > 0:
        print(f"[VALIDATION] {n_dropped} titik sumur di-drop (jarak > {max_distance}m dari grid MT).")

    return df_result


def calculate_cluster_statistics(
    df_well_validated: pd.DataFrame,
    fracture_col: str = "P21_CONDUCTIVE_FRACTURE",
) -> pd.DataFrame:
    """Menghitung statistik deskriptif P21 (densitas rekahan) per klaster
    yang dilewati sumur bor.
    """
    print(f"[VALIDATION] Menghitung statistik per klaster pada kolom '{fracture_col}'...")

    stats = (
        df_well_validated
        .groupby("assigned_cluster")[fracture_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    stats.columns = ["Cluster_Label", "Jumlah_Titik_Log", "Rata_Rata_P21", "Standar_Deviasi", "Min_P21", "Max_P21"]

    return stats
