"""mt_transform.py — Transformasi domain data MT (Magnetotellurik)."""

import pandas as pd
from ..utils import log10_safe


def transform_mt_log_res(df_mt: pd.DataFrame, resistivity_col: str = "Resistivity") -> pd.DataFrame:
    """Mengubah nilai resistivitas linier menjadi skala logaritmik (Log10).

    Baris dengan resistivitas <= 0 (inversi artifact) di-drop setelah transformasi.
    """
    print("[CONDITIONING] Melakukan transformasi Log10 pada resistivitas MT...")
    df = df_mt.copy()
    df["log_res"] = log10_safe(df[resistivity_col].values)

    # Drop baris yang menghasilkan NaN (nilai <= 0 dari inversi artifact)
    n_before = len(df)
    df = df.dropna(subset=["log_res"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[CONDITIONING] {n_dropped} baris di-drop karena resistivitas <= 0.")

    return df
