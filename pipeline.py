"""pipeline.py — Entry point utama. Orkestrasi seluruh alur ML dari raw data ke validasi.

Jalankan: python pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from io import load_mt_data, load_earthquake_catalog, load_well_log, save_processed_grid
from conditioning import transform_mt_log_res, convert_eq_coordinates, map_points_to_voxel_grid
from clustering_model import scale_features, run_hdbscan
from well_validation import extract_cluster_along_well, calculate_cluster_statistics
from plots import plot_2d_slice_with_eq
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# KONFIGURASI — sesuaikan path dan parameter di sini
# ---------------------------------------------------------------------------
CONFIG = {
    # Path data input
    "mt_filepath":        "data/raw/mt_inversion_3d.xyz",
    "eq_filepath":        "data/raw/usgs_earthquake_catalog.csv",
    "well_filepath":      "data/raw/well_log.las",

    # Path output
    "grid_output":        "data/processed/mt_grid_conditioned.csv",

    # Parameter conditioning
    "utm_zone":           12,
    "hemisphere":         "N",
    "eq_radius_meters":   100,

    # Elevasi filter (meter)
    "z_min":              -3000,
    "z_max":              1000,

    # Fitur ML
    "feature_cols":       ["X", "Y", "Z", "log_res", "seismic_density"],
    "spatial_weight":     0.5,

    # Parameter HDBSCAN
    "min_cluster_size":   100,
    "min_samples":        10,

    # Plot
    "slice_y_coord":      500_000,   # meter UTM
    "slice_tolerance":    50,
}

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("PIPELINE MT-HDBSCAN PERMEABLE PATHWAY DETECTION")
    print("=" * 60)

    # 1. Load raw data
    df_mt  = load_mt_data(CONFIG["mt_filepath"])
    df_eq  = load_earthquake_catalog(CONFIG["eq_filepath"])
    df_well = load_well_log(CONFIG["well_filepath"])

    # 2. Filter elevasi grid MT
    df_mt = df_mt[df_mt["Z"].between(CONFIG["z_min"], CONFIG["z_max"])].copy()
    print(f"[PIPELINE] Grid MT setelah filter elevasi: {len(df_mt):,} voxel.")

    # 3. Conditioning
    df_mt  = transform_mt_log_res(df_mt)
    df_eq  = convert_eq_coordinates(df_eq, utm_zone=CONFIG["utm_zone"], hemisphere=CONFIG["hemisphere"])
    df_mt  = map_points_to_voxel_grid(df_mt, df_eq, radius_meters=CONFIG["eq_radius_meters"])

    # 4. Simpan grid hasil conditioning (checkpoint)
    save_processed_grid(df_mt, CONFIG["grid_output"])

    # 5. Scaling & HDBSCAN
    X_scaled, _ = scale_features(df_mt, CONFIG["feature_cols"], CONFIG["spatial_weight"])
    labels, clusterer = run_hdbscan(X_scaled, CONFIG["min_cluster_size"], CONFIG["min_samples"])
    df_mt["cluster_labels"] = labels

    # 6. Validasi sumur
    df_well_validated = extract_cluster_along_well(df_mt, df_well)
    stats = calculate_cluster_statistics(df_well_validated)
    print("\n[PIPELINE] Statistik validasi sumur:")
    print(stats.to_string(index=False))

    # 7. Plot
    global_mag_min = df_eq["magnitude"].min()
    fig = plot_2d_slice_with_eq(
        df_mt, df_eq,
        slice_coord=CONFIG["slice_y_coord"],
        tolerance=CONFIG["slice_tolerance"],
        global_mag_min=global_mag_min,
    )
    plt.show()

    print("\n[PIPELINE] Selesai.")


if __name__ == "__main__":
    run()
