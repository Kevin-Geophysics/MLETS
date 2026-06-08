"""io.py — Semua operasi baca/tulis file. Tidak ada logika bisnis di sini."""

import os
import lasio
import pandas as pd


def load_mt_data(filepath: str) -> pd.DataFrame:
    """Membaca file inversi 3D MT (.xyz atau .dat).

    Mengasumsikan kolom dipisahkan whitespace/tab dengan header: X, Y, Z, Resistivity.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File MT tidak ditemukan: {filepath}")

    print(f"[IO] Membaca data 3D MT dari: {os.path.basename(filepath)}")
    return pd.read_csv(filepath, sep=r"\s+")


def load_earthquake_catalog(filepath: str) -> pd.DataFrame:
    """Membaca katalog gempa USGS format .csv."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Katalog gempa tidak ditemukan: {filepath}")

    print(f"[IO] Membaca katalog gempa dari: {os.path.basename(filepath)}")
    return pd.read_csv(filepath)


def load_well_log(filepath: str) -> pd.DataFrame:
    """Membaca file log sumur .las (data petrophysics/fracture).

    Depth/DEPT di-reset menjadi kolom biasa (bukan index) agar konsisten
    dengan ekspektasi well_validation.py yang membutuhkan kolom X, Y, Z eksplisit.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File LAS tidak ditemukan: {filepath}")

    print(f"[IO] Membaca file LAS dari: {os.path.basename(filepath)}")
    las = lasio.read(filepath)
    return las.df().reset_index()  # DEPTH jadi kolom, bukan index


def load_well_trajectory(filepath: str) -> pd.DataFrame:
    """Membaca file lintasan sumur bor (directional survey) format .csv atau .xlsx.

    Berbeda dengan load_well_log (.las), trajectory hanya berisi koordinat
    jalur bor: X, Y, Z TVD — tidak butuh lasio.

    Kolom yang diharapkan: X, Y, Z TVD (dalam US survey feet).
    Konversi ke UTM meter dilakukan di conditioning/well_transform.py.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File well trajectory tidak ditemukan: {filepath}")

    ext = os.path.splitext(filepath)[-1].lower()
    print(f"[IO] Membaca well trajectory dari: {os.path.basename(filepath)}")

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    elif ext == ".csv":
        return pd.read_csv(filepath)
    else:
        # Coba whitespace-separated sebagai fallback (.txt, .dat, dll.)
        return pd.read_csv(filepath, sep=r"\s+")


def save_processed_grid(df: pd.DataFrame, output_path: str) -> None:
    """Menyimpan DataFrame hasil akhir ke .csv."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[IO] Data disimpan ke: {output_path}")
