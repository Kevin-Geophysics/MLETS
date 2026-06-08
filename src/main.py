"""main.py — Entry point pipeline. Bisa dijalankan langsung atau di-import dari notebook.

CLI  : python main.py
NB   : from main import run_pipeline; run_pipeline(CONFIG)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src import *
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# KONFIGURASI — sesuaikan path dan parameter di sini
# ---------------------------------------------------------------------------
CONFIG = {
    # --- Path data input ---
    "mt_filepath":          "data/raw/mt_inversion_3d.xyz",
    "eq_filepath":          "data/raw/usgs_earthquake_catalog.csv",
    "well_log_filepath":    "data/raw/well_log.las",           # petrophysics (.las)
    "well_traj_filepath":   "data/raw/well_trajectory.csv",    # directional survey (X, Y, Z TVD)

    # --- Path output ---
    "grid_output":          "data/processed/mt_grid_conditioned.csv",

    # --- Wellhead info (dari directional survey report) ---
    "wellhead_x_utm":       335450.9891,   # meter, UTM absolut
    "wellhead_y_utm":       4263037.906,   # meter, UTM absolut
    "wellhead_elevation_usft": 5050.0,     # usft — cek header LAS atau survey report

    # --- Parameter conditioning ---
    "utm_zone":             12,
    "hemisphere":           "N",
    "eq_radius_meters":     100,

    # --- Filter elevasi grid MT (meter) ---
    "z_min":                -3000,
    "z_max":                1000,

    # --- Fitur ML ---
    "feature_cols":         ["X", "Y", "Z", "log_res", "seismic_density"],
    "spatial_weight":       0.5,

    # --- Parameter HDBSCAN ---
    "min_cluster_size":     100,
    "min_samples":          10,

    # --- Plot ---
    "slice_y_coord":        4263037.0,   # UTM Northing sumur (meter)
    "slice_tolerance":      50,
}


def run_pipeline(config: dict = CONFIG) -> dict:
    """Jalankan full pipeline. Mengembalikan dict berisi semua objek hasil
    agar bisa diakses langsung di notebook setelah pemanggilan.

    Returns
    -------
    dict dengan keys: df_mt, df_eq, df_well_log, df_well_traj, stats, fig
    """
    print("=" * 60)
    print("PIPELINE MT-HDBSCAN — PERMEABLE PATHWAY DETECTION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------
    df_mt       = load_mt_data(config["mt_filepath"])
    df_eq       = load_earthquake_catalog(config["eq_filepath"])
    df_well_log = load_well_log(config["well_log_filepath"])
    df_well_raw = load_well_trajectory(config["well_traj_filepath"])

    # ------------------------------------------------------------------
    # 2. FILTER ELEVASI GRID MT
    # ------------------------------------------------------------------
    df_mt = df_mt[df_mt["Z"].between(config["z_min"], config["z_max"])].copy()
    print(f"[MAIN] Grid MT setelah filter elevasi: {len(df_mt):,} voxel.")

    # ------------------------------------------------------------------
    # 3. CONDITIONING
    # ------------------------------------------------------------------
    df_mt   = transform_mt_log_res(df_mt)
    df_eq   = convert_eq_coordinates(df_eq, utm_zone=config["utm_zone"],
                                     hemisphere=config["hemisphere"])
    df_mt   = map_points_to_voxel_grid(df_mt, df_eq,
                                       radius_meters=config["eq_radius_meters"])

    # Konversi trajectory: X/Y/Z TVD (usft) → UTM meter
    df_well_traj = convert_well_trajectory(
        df_well_raw,
        wellhead_x_utm=config["wellhead_x_utm"],
        wellhead_y_utm=config["wellhead_y_utm"],
        wellhead_elevation_usft=config["wellhead_elevation_usft"],
    )

    # Simpan checkpoint grid
    save_processed_grid(df_mt, config["grid_output"])

    # ------------------------------------------------------------------
    # 4. CLUSTERING
    # ------------------------------------------------------------------
    X_scaled, _ = scale_features(df_mt, config["feature_cols"],
                                  config["spatial_weight"])
    labels, _   = run_hdbscan(X_scaled, config["min_cluster_size"],
                               config["min_samples"])
    df_mt["cluster_labels"] = labels

    # ------------------------------------------------------------------
    # 5. VALIDASI SUMUR LOG
    # ------------------------------------------------------------------
    df_well_validated = extract_cluster_along_well(df_mt, df_well_log)
    stats = calculate_cluster_statistics(df_well_validated)
    print("\n[MAIN] Statistik validasi sumur:")
    print(stats.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. PLOT
    # ------------------------------------------------------------------
    fig = plot_2d_slice_with_eq(
        df_mt, df_eq,
        df_well_trajectory=df_well_traj,
        slice_coord=config["slice_y_coord"],
        tolerance=config["slice_tolerance"],
        global_mag_min=df_eq["magnitude"].min(),
    )
    plt.show()

    print("\n[MAIN] Pipeline selesai.")

    return {
        "df_mt":         df_mt,
        "df_eq":         df_eq,
        "df_well_log":   df_well_log,
        "df_well_traj":  df_well_traj,
        "stats":         stats,
        "fig":           fig,
    }


if __name__ == "__main__":
    run_pipeline()
