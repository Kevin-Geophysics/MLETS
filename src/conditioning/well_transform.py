"""well_transform.py — Preprocessing koordinat data lintasan sumur bor.

Konversi dari sistem koordinat raw directional survey (US survey feet, lokal)
ke koordinat UTM meter absolut yang konsisten dengan grid MT.
"""

import numpy as np
import pandas as pd
from utils import usft_to_meters, well_tvd_to_z_meters


def convert_well_trajectory(
    df_well: pd.DataFrame,
    wellhead_x_utm: float,
    wellhead_y_utm: float,
    wellhead_elevation_usft: float,
    x_col: str = "X",
    y_col: str = "Y",
    tvd_col: str = "Z TVD",
) -> pd.DataFrame:
    """Mengonversi koordinat lintasan sumur dari US survey feet (lokal) ke
    UTM meter absolut, siap di-overlay ke grid MT.

    Alur konversi:
        X/Y raw (usft, relatif) → usft_to_meters → + wellhead UTM absolut
        Z TVD (usft, positif bawah) → koreksi elevasi → meter, negatif bawah

    Parameters
    ----------
    wellhead_x_utm, wellhead_y_utm : float
        Koordinat UTM absolut wellhead dalam meter.
        Untuk Utah FORGE: X=335450.9891, Y=4263037.906
    wellhead_elevation_usft : float
        Elevasi permukaan wellhead (usft) untuk koreksi Z.
        Sesuai catatan "Revised Z TVD to be Survey - Elevation".
    x_col, y_col : str
        Nama kolom X dan Y di df_well (default: "X", "Y").
    tvd_col : str
        Nama kolom TVD di df_well (default: "Z TVD").

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom tambahan:
        - X_utm_well  : Easting UTM absolut (meter)
        - Y_utm_well  : Northing UTM absolut (meter)
        - Z_meters_well : Kedalaman dalam meter, negatif ke bawah
    """
    print("[CONDITIONING] Mengonversi koordinat lintasan sumur ke UTM meter...")
    df = df_well.copy()

    # Konversi offset X/Y (usft relatif) → meter absolut
    # X/Y di data adalah offset dari wellhead dalam usft,
    # lalu ditambah posisi absolut wellhead UTM
    df["X_utm_well"] = usft_to_meters(df[x_col].values) + wellhead_x_utm
    df["Y_utm_well"] = usft_to_meters(df[y_col].values) + wellhead_y_utm

    # Konversi TVD → Z meter dengan koreksi elevasi
    df["Z_meters_well"] = well_tvd_to_z_meters(
        tvd_usft=df[tvd_col].values,
        elevation_usft=wellhead_elevation_usft,
    )

    print(
        f"[CONDITIONING] Well trajectory: {len(df)} titik, "
        f"Z range: {df['Z_meters_well'].min():.1f} ~ {df['Z_meters_well'].max():.1f} m"
    )
    return df
