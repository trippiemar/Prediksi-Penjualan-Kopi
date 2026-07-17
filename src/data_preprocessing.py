"""
data_preprocessing.py
----------------------
Script untuk membersihkan (cleaning), menggabungkan (merge), dan melakukan
feature engineering pada dataset penjualan kafe (Kaggle: dirty_cafe_sales.csv)
dan data inflasi bulanan Bank Indonesia (Data_Inflasi.xlsx).

Problem statement: Prediksi Total Spent (nilai transaksi) sebuah transaksi
kafe berdasarkan atribut transaksi (item, quantity, price per unit, payment
method, location, waktu transaksi) dengan mempertimbangkan faktor
makroekonomi (tingkat inflasi bulanan).

Output:
    data/processed/cafe_sales_clean.csv
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

BULAN_INDO_KE_INGGRIS = {
    "Januari": "January", "Februari": "February", "Maret": "March", "April": "April",
    "Mei": "May", "Juni": "June", "Juli": "July", "Agustus": "August",
    "September": "September", "Oktober": "October", "November": "November", "Desember": "December",
}


def load_cafe_sales(path=None) -> pd.DataFrame:
    path = path or os.path.join(RAW_DIR, "dirty_cafe_sales.csv")
    df = pd.read_csv(path)
    return df


def load_inflasi(path=None) -> pd.DataFrame:
    path = path or os.path.join(RAW_DIR, "Data_Inflasi.xlsx")
    df = pd.read_excel(path, skiprows=4)
    df.columns = [str(c).strip() for c in df.columns]

    df["Periode_Clean"] = df["Periode"].astype(str).str.strip()
    for indo, eng in BULAN_INDO_KE_INGGRIS.items():
        df["Periode_Clean"] = df["Periode_Clean"].str.replace(indo, eng, case=False)

    df["Tanggal"] = pd.to_datetime(df["Periode_Clean"], format="%B %Y", errors="coerce")
    df = df.dropna(subset=["Tanggal"])

    df = df.rename(columns={"Data Inflasi": "Inflasi"})
    df["Inflasi"] = (
        df["Inflasi"].astype(str).str.replace("%", "").str.strip()
    )
    df["Inflasi"] = pd.to_numeric(df["Inflasi"], errors="coerce")
    df["Bulan_Tahun"] = df["Tanggal"].dt.to_period("M")
    return df[["Bulan_Tahun", "Inflasi"]]


def clean_cafe_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan placeholder ERROR/UNKNOWN, tipe data, dan missing values."""
    df = df.copy()

    # Ganti placeholder string jadi NaN supaya bisa ditangani konsisten
    df = df.replace(["ERROR", "UNKNOWN", "nan", "NaN"], np.nan)

    # Konversi tipe data numerik & tanggal
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
    df["Total Spent"] = pd.to_numeric(df["Total Spent"], errors="coerce")
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

    # Buang baris tanpa tanggal transaksi (tidak bisa dipakai untuk fitur waktu/inflasi)
    df = df.dropna(subset=["Transaction Date"])

    # Buang baris tanpa Item (kunci utama produk)
    df = df.dropna(subset=["Item"])

    # Quantity & Price Per Unit wajib ada untuk hitung ulang Total Spent
    df = df.dropna(subset=["Quantity", "Price Per Unit"])

    # Rekonstruksi Total Spent yang hilang dari Quantity * Price
    df["Total Spent"] = df["Total Spent"].fillna(df["Quantity"] * df["Price Per Unit"])

    # Kategorikal: isi missing dengan label "Unknown" (kategori tersendiri, bukan dibuang)
    df["Payment Method"] = df["Payment Method"].fillna("Unknown").astype(str).str.strip()
    df["Location"] = df["Location"].fillna("Unknown").astype(str).str.strip()

    # Buang duplikat transaksi (berdasarkan Transaction ID)
    df = df.drop_duplicates(subset=["Transaction ID"])

    # Buang outlier ekstrem pakai IQR pada Total Spent
    q1, q3 = df["Total Spent"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[(df["Total Spent"] >= lower) & (df["Total Spent"] <= upper)]

    df = df.sort_values("Transaction Date").reset_index(drop=True)
    return df


def feature_engineering(df: pd.DataFrame, df_inflasi: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Bulan_Tahun"] = df["Transaction Date"].dt.to_period("M")
    df = df.merge(df_inflasi, on="Bulan_Tahun", how="left")
    df["Inflasi"] = df["Inflasi"].fillna(df["Inflasi"].median())

    # Fitur waktu
    df["Year"] = df["Transaction Date"].dt.year
    df["Month"] = df["Transaction Date"].dt.month
    df["DayOfWeek"] = df["Transaction Date"].dt.dayofweek  # 0=Senin
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    df = df.drop(columns=["Bulan_Tahun", "Transaction ID"])
    return df


def run_pipeline(save_path=None) -> pd.DataFrame:
    raw = load_cafe_sales()
    inflasi = load_inflasi()
    clean = clean_cafe_sales(raw)
    final_df = feature_engineering(clean, inflasi)

    save_path = save_path or os.path.join(PROCESSED_DIR, "cafe_sales_clean.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_df.to_csv(save_path, index=False)
    print(f"Selesai. Data bersih disimpan di: {save_path}")
    print(f"Jumlah baris awal : {len(raw)}")
    print(f"Jumlah baris bersih: {len(final_df)}")
    return final_df


if __name__ == "__main__":
    run_pipeline()
