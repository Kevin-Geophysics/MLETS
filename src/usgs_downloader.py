from io import StringIO
import pandas as pd
import requests


def usgs_downloader(bbox, min_mag, max_mag):
    """Men-download data gempa USGS berdasarkan dictionary bounding box (bbox)

    serta batas minimum dan maksimum magnitudo yang eksplisit.
    """
    print("[USGS DOWNLOAD] Menghubungi server USGS Earthquake API...")

    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # Menyusun parameter dengan rapi dan presisi
    params = {
        "format": "csv",
        "starttime": bbox["start_time"],
        "endtime": bbox["end_time"],
        "minlatitude": bbox["min_lat"],
        "maxlatitude": bbox["max_lat"],
        "minlongitude": bbox["min_lon"],
        "maxlongitude": bbox["max_lon"],
        "minmagnitude": min_mag,
        "maxmagnitude": max_mag,
        "orderby": "time-asc",
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        print(
            f"[USGS DOWNLOAD] Sukses! Terunduh {len(df):,} event gempa mikro (Mw {min_mag} s.d {max_mag})."
        )
        return df
    else:
        raise Exception(
            f"Gagal mengambil data. Status Code: {response.status_code}"
        )