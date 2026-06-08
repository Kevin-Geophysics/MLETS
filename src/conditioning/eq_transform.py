"""eq_transform.py — Transformasi domain data katalog gempa (USGS)."""

import pandas as pd
from ..utils import latlon_to_utm, depth_km_to_z_meters


def convert_eq_coordinates(
    df_eq: pd.DataFrame,
    utm_zone: int = 12,
    hemisphere: str = "N",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    depth_col: str = "depth",
) -> pd.DataFrame:
    """Mengonversi koordinat gempa USGS (WGS84 Lat/Lon) ke UTM meter dan
    menyamakan konvensi tanda Z (kedalaman) dengan grid MT.

    Parameters
    ----------
    utm_zone : int
        Nomor zona UTM. Utah FORGE / Idaho umumnya Zone 12.
    hemisphere : str
        'N' untuk Northern, 'S' untuk Southern Hemisphere.
    """
    print(f"[CONDITIONING] Mengonversi koordinat gempa ke UTM Zone {utm_zone}{hemisphere}...")
    df = df_eq.copy()

    # Konversi Lat/Lon → UTM (delegasi ke utils)
    df["X_utm"], df["Y_utm"] = latlon_to_utm(
        lon=df[lon_col].values,
        lat=df[lat_col].values,
        utm_zone=utm_zone,
        hemisphere=hemisphere,
    )

    # Konversi depth km → Z meter negatif (delegasi ke utils)
    df["Z_meters"] = depth_km_to_z_meters(df[depth_col].values)

    return df
