# MT-HDBSCAN Permeable Pathway Detection

Deteksi zona permeabel bawah permukaan menggunakan unsupervised machine learning (HDBSCAN)
pada data inversi 3D Magnetotellurik (MT), katalog gempa USGS, dan log sumur bor.

## Struktur Project

```
project/
├── data/
│   ├── raw/          # Data mentah (tidak di-push ke GitHub)
│   └── processed/    # Hasil conditioning (tidak di-push ke GitHub)
├── notebook/         # Eksplorasi & visualisasi Jupyter
├── output/           # Gambar, tabel hasil akhir
├── referensi/        # Paper, dokumen pendukung
├── example/          # Contoh data kecil untuk testing
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── io.py
│   ├── clustering_model.py
│   ├── well_validation.py
│   ├── conditioning/
│   │   ├── __init__.py
│   │   ├── mt_transform.py
│   │   ├── eq_transform.py
│   │   └── grid_alignment.py
│   └── plots/
│       ├── __init__.py
│       └── cross_section.py
├── pipeline.py
├── requirements.txt
└── .gitignore
```

## Instalasi

```bash
git clone https://github.com/username/repo-name.git
cd repo-name
pip install -r requirements.txt
```

## Cara Pakai

1. Taruh data mentah di `data/raw/`
2. Sesuaikan `CONFIG` di `pipeline.py`
3. Jalankan pipeline:

```bash
python pipeline.py
```

## Alur Pipeline

```
Raw Data (MT .xyz, USGS .csv, Well .las)
    ↓ io.py
Load & Validate
    ↓ conditioning/
Log10 Resistivity → Koordinat UTM → Grid Alignment (KDTree)
    ↓ clustering_model.py
Scale Features → HDBSCAN
    ↓ well_validation.py
Validasi Silang dengan Log Sumur
    ↓ plots/
Visualisasi Penampang 2D
```
