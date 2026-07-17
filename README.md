# ☕ Capstone Project: Prediksi Estimasi Nilai Belanja Kafe dengan Faktor Inflasi

Capstone Project untuk UAS Mata Kuliah **Pembelajaran Mesin** — Universitas Dian
Nuswantoro, Fakultas Ilmu Komputer, Genap 2025/2026.

**Live App:** (https://prediksi-penjualan-kopi-mswkczpwipfrxqjgctsqz7.streamlit.app)
**Video Presentasi:** _(isi link YouTube di sini)_

---

## 1. Problem Statement

Sebuah kafe ingin memperkirakan **estimasi nilai belanja (`Total Spent`)** dari
sebuah transaksi hanya berdasarkan **pilihan menu & konteks transaksi** (waktu,
lokasi, metode pembayaran) — tanpa perlu menunggu quantity/harga final diketahui —
sekaligus mempertimbangkan **inflasi bulanan** sebagai faktor makroekonomi eksternal.

Estimasi ini berguna untuk **forecasting pendapatan** dan **perencanaan operasional**
(stok bahan baku, staffing) secara real-time.

**Metrik kesuksesan:** RMSE serendah mungkin pada test set, dengan model yang tetap
dapat diinterpretasikan oleh stakeholder non-teknis (pemilik kafe).

## 2. Sumber Data

| Dataset                | Sumber                            | Deskripsi                                                                         |
| ---------------------- | --------------------------------- | --------------------------------------------------------------------------------- |
| `dirty_cafe_sales.csv` | [Kaggle](https://www.kaggle.com/) | 10.000 transaksi kafe (sengaja kotor: `ERROR`/`UNKNOWN`, missing value, duplikat) |
| `Data_Inflasi.xlsx`    | Bank Indonesia                    | Data inflasi bulanan (%), tahun 2023                                              |

Setelah proses cleaning, tersisa ± 7.500 transaksi bersih yang dipakai untuk modeling.

> ⚠️ **Catatan penting soal desain fitur:** `Total Spent = Quantity x Price Per Unit`
> secara matematis persis. Kedua kolom tersebut **sengaja tidak dipakai sebagai
> fitur** model karena akan menyebabkan _data leakage_ (model jadi trivial, R²=1.0).
> Fitur yang dipakai: `Item`, `Payment Method`, `Location`, `Inflasi`, dan fitur
> waktu (`Year`, `Month`, `DayOfWeek`, `IsWeekend`).

## 3. Metodologi

```
Data Acquisition -> Cleaning -> Feature Engineering -> Modeling & Tuning
      -> Evaluation -> Interpretation (SHAP/Permutation Importance) -> Deployment
```

- **Model 1:** Linear Regression (baseline, interpretable)
- **Model 2:** Random Forest Regressor (ensemble, di-tuning dengan `GridSearchCV`)
- **Evaluasi:** R², MAE, MSE, RMSE, residual analysis
- **Interpretasi:** Permutation Importance & SHAP TreeExplainer

## 4. Hasil Ringkas

| Model             | R² (Test) | MAE (Test) | RMSE (Test) |
| ----------------- | --------- | ---------- | ----------- |
| Linear Regression | ~0.40     | ~3.44      | ~4.24       |
| Random Forest     | ~0.39     | ~3.49      | ~4.29       |

Model terbaik: **Linear Regression** (RMSE terendah + paling interpretable).
Lihat `reports/model_comparison.csv` dan `notebooks/02_modeling.ipynb` untuk detail.

## 5. Struktur Repository

```
capstone-project-data-mining/
│
├── data/
│   ├── raw/                  # Data mentah (csv & xlsx asli)
│   ├── processed/            # Data yang sudah dibersihkan
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb              # EDA dan preprocessing
│   ├── 02_modeling.ipynb         # Pemodelan dan evaluasi
│   └── 03_interpretation.ipynb   # Interpretasi model (SHAP)
├── src/
│   ├── data_preprocessing.py # Script cleaning + feature engineering
│   ├── train_model.py        # Script training & tuning model
│   ├── evaluate_model.py     # Script evaluasi & visualisasi
│   └── utils.py              # Fungsi utilitas bersama
├── models/
│   ├── best_model.pkl        # Model terbaik
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── metadata.json         # Info fitur, target, hasil evaluasi semua model
├── app/
│   ├── app.py                 # Aplikasi Streamlit utama
│   ├── pages/                 # (opsional) halaman tambahan
│   └── assets/                # Gambar/CSS pendukung
├── reports/
│   ├── figures/                # Semua visualisasi evaluasi & interpretasi
│   ├── feature_importance.csv
│   ├── residuals_test.csv
│   ├── model_comparison.csv
│   └── final_report.pdf        # Laporan teknis (isi manual/export)
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Cara Menjalankan Secara Lokal

```bash
git clone <repo-url>
cd capstone-project-data-mining

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 1. Bersihkan data & gabungkan dengan data inflasi
python src/data_preprocessing.py

# 2. Latih & tuning model, simpan pickle
python src/train_model.py

# 3. Generate visualisasi evaluasi
python src/evaluate_model.py

# 4. Jalankan aplikasi
streamlit run app/app.py
```

## 7. Author

Yunanda Mario Putra — A11.2024.15923 — Universitas Dian Nuswantoro
