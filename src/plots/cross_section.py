"""cross_section.py — Plot penampang 2D slice hasil HDBSCAN + overlay gempa + sumur."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FT_TO_M = 0.3048


def _load_well(
    csv_path: str,
    wellhead_x: float,
    wellhead_y: float,
    wellhead_elev_m: float = 0.0,
) -> pd.DataFrame:
    """Baca CSV sumur (kolom N, E, Z dalam ft), konversi ke UTM meter.

    Returns DataFrame dengan kolom X, Y, Z (meter, Z negatif ke bawah).
    """
    df = pd.read_csv(csv_path)
    df_out = pd.DataFrame()
    df_out["X"] = wellhead_x + df["E"] * FT_TO_M
    df_out["Y"] = wellhead_y + df["N"] * FT_TO_M
    df_out["Z"] = wellhead_elev_m - df["Z"] * FT_TO_M  # TVD positif → Z negatif
    return df_out


def plot_2d_slice_with_eq(
    df_clustered: pd.DataFrame,
    df_eq_utm: pd.DataFrame,
    slice_coord: float,
    tolerance: float = 50,
    slice_by: str = "Y",
    x_axis: str = "X",
    z_axis: str = "Z",
    eq_x_col: str = "X_utm",
    eq_y_col: str = "Y_utm",
    eq_z_col: str = "Z_meters",
    eq_mag_col: str = "magnitude",
    global_mag_min: float | None = None,
    wells: list[dict] | None = None,
) -> tuple[plt.Figure, list[str]]:
    """Penampang 2D klaster HDBSCAN dengan overlay distribusi gempa dan sumur bor.

    Parameters
    ----------
    wells : list of dict, optional
        Setiap dict berisi:
        - 'path'        : str, path ke CSV sumur
        - 'name'        : str, label sumur
        - 'wellhead_x'  : float, UTM X wellhead (meter)
        - 'wellhead_y'  : float, UTM Y wellhead (meter)
        - 'wellhead_elev_m' : float, elevasi wellhead (meter), default 0

    Returns
    -------
    fig : plt.Figure
    out_of_range : list[str]
        Nama sumur yang di luar range slice.
    """
    print(f"[PLOT] Penampang 2D slice {slice_by}={slice_coord}m...")

    # ------------------------------------------------------------------
    # 1. FILTER SLICE
    # ------------------------------------------------------------------
    df_grid = df_clustered[
        df_clustered[slice_by].between(slice_coord - tolerance, slice_coord + tolerance)
    ].copy()

    eq_filter_col = eq_y_col if slice_by == "Y" else eq_x_col
    df_eq = df_eq_utm[
        df_eq_utm[eq_filter_col].between(slice_coord - tolerance, slice_coord + tolerance)
    ].copy()

    # ------------------------------------------------------------------
    # 2. PLOT
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))

    # Noise / host rock
    df_noise = df_grid[df_grid["cluster_labels"] == -1]
    ax.scatter(df_noise[x_axis], df_noise[z_axis],
               c="lightgray", s=15, label="Noise / Host Rock (MT)", alpha=0.3)

    # Klaster valid
    df_valid = df_grid[df_grid["cluster_labels"] != -1]
    if not df_valid.empty:
        sc_grid = ax.scatter(
            df_valid[x_axis], df_valid[z_axis],
            c=df_valid["cluster_labels"], cmap="tab20",
            s=35, edgecolors="none", label="HDBSCAN Clusters", alpha=0.7,
        )
        cbar = fig.colorbar(sc_grid, ax=ax, label="Cluster ID", pad=0.02, location="right")
        cbar.set_ticks(np.unique(df_valid["cluster_labels"]))

    # Overlay gempa
    if not df_eq.empty:
        mag_min = global_mag_min if global_mag_min is not None else df_eq[eq_mag_col].min()
        mag_max = df_eq[eq_mag_col].max()
        sizes = (df_eq[eq_mag_col] - mag_min + 1.5) ** 3 * 2

        eq_plot_x_col = eq_y_col if slice_by == "X" else eq_x_col

        sc_eq = ax.scatter(
            df_eq[eq_plot_x_col], df_eq[eq_z_col],
            c=df_eq[eq_mag_col], cmap="YlOrRd",
            s=sizes, edgecolors="black", linewidths=0.8,
            label="Earthquake Events (USGS)", alpha=0.9, zorder=5,
        )
        fig.colorbar(sc_eq, ax=ax, label="Earthquake Magnitude (Mw)", pad=0.08, location="right")

        mag_samples = np.linspace(mag_min, mag_max, 3)
        for mag_sample in mag_samples:
            ax.scatter([], [], c="darkred", alpha=0.7,
                       s=(mag_sample - mag_min + 1.5) ** 3 * 2,
                       edgecolors="black", label=f"Mw {mag_sample:.1f}")

    # ------------------------------------------------------------------
    # 3. OVERLAY SUMUR
    # ------------------------------------------------------------------
    out_of_range = []

    if wells:
        colors_well = ["black", "deepskyblue", "lime", "magenta", "yellow"]
        for i, well in enumerate(wells):
            elev = well.get("wellhead_elev_m", 0.0)
            df_w = _load_well(well["path"], well["wellhead_x"], well["wellhead_y"], elev)

            # Cek apakah sumur masuk range slice
            check_col = "X" if slice_by == "X" else "Y"
            in_range = df_w[check_col].between(slice_coord - tolerance, slice_coord + tolerance)

            if in_range.sum() == 0:
                out_of_range.append(well["name"])
                print(f"[PLOT] Sumur '{well['name']}' di luar range slice, tidak diplot.")
                continue

            df_w_slice = df_w[in_range]
            plot_x_well = "Y" if slice_by == "X" else "X"
            color = colors_well[i % len(colors_well)]

            ax.plot(df_w_slice[plot_x_well], df_w_slice["Z"],
                    color=color, linewidth=2, zorder=10, label=well["name"])
            # Marker wellhead (titik pertama)
            ax.scatter(df_w_slice[plot_x_well].iloc[0], df_w_slice["Z"].iloc[0],
                       color=color, s=80, zorder=11, marker="^")

    # ------------------------------------------------------------------
    # 4. DEKORASI
    # ------------------------------------------------------------------
    ax.set_title(
        f"Penampang 2D Permeable Pathway vs Distribusi Gempa\n(Slice {slice_by}: {slice_coord} m)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel(f"Koordinat {x_axis} (meter)", fontsize=11)
    ax.set_ylabel(f"Kedalaman {z_axis} (meter)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", scatterpoints=1, fontsize=10)
    plt.tight_layout()

    return fig, out_of_range