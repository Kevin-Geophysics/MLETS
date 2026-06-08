"""clustering_model.py — Scaling fitur dan eksekusi HDBSCAN."""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler


def scale_features(
    df: pd.DataFrame,
    feature_cols: list,
    spatial_weight: float = 1.0,
) -> tuple[np.ndarray, StandardScaler]:
    """Standardisasi (Z-score) fitur input ML.

    Parameters
    ----------
    spatial_weight : float
        Faktor pengali untuk kolom X, Y, Z setelah di-scale.
        Set < 1.0 (misal 0.5) agar model lebih mementingkan kemiripan
        properti batuan daripada kedekatan posisi geografis.
    """
    print("[MODEL] Standardisasi fitur dengan StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    if spatial_weight != 1.0:
        print(f"[MODEL] Spatial weight {spatial_weight} diterapkan pada kolom X, Y, Z...")
        for col in ["X", "Y", "Z"]:
            if col in X_scaled_df.columns:
                X_scaled_df[col] *= spatial_weight

    return X_scaled_df.values, scaler


def run_hdbscan(
    X_scaled: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """Menjalankan HDBSCAN pada data yang sudah di-scale.

    Returns
    -------
    labels : np.ndarray
        Label klaster per titik. -1 = noise.
    clusterer : hdbscan.HDBSCAN
        Objek fitted, tersimpan untuk keperluan plot hierarki atau soft clustering.
    """
    print(f"[MODEL] HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"[MODEL] Hasil: {n_clusters} klaster ditemukan, {n_noise} titik noise.")

    return labels, clusterer
