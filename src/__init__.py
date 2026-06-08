"""src/__init__.py — Single entry point untuk seluruh modul pipeline.

Usage:
    from src import *

Semua fungsi publik langsung tersedia tanpa perlu import per-modul.
"""

# I/O
from .io import (
    load_mt_data,
    load_earthquake_catalog,
    load_well_log,
    save_processed_grid,
)

# Utilities (pure functions)
from .utils import (
    sphere_volume,
    seismic_density,
    log10_safe,
    latlon_to_utm,
    depth_km_to_z_meters,
)

# Conditioning
from .conditioning import (
    transform_mt_log_res,
    convert_eq_coordinates,
    map_points_to_voxel_grid,
)

# ML Model
from .clustering_model import (
    scale_features,
    run_hdbscan,
)

# Validation
from .well_validation import (
    extract_cluster_along_well,
    calculate_cluster_statistics,
)

# Plots
from .plots import (
    plot_2d_slice_with_eq,
)

__all__ = [
    # io
    "load_mt_data",
    "load_earthquake_catalog",
    "load_well_log",
    "save_processed_grid",
    # utils
    "sphere_volume",
    "seismic_density",
    "log10_safe",
    "latlon_to_utm",
    "depth_km_to_z_meters",
    # conditioning
    "transform_mt_log_res",
    "convert_eq_coordinates",
    "map_points_to_voxel_grid",
    # model
    "scale_features",
    "run_hdbscan",
    # validation
    "extract_cluster_along_well",
    "calculate_cluster_statistics",
    # plots
    "plot_2d_slice_with_eq",
]
