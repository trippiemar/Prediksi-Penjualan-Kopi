"""
app.py - Aplikasi Streamlit Capstone Project Pembelajaran Mesin
Prediksi Estimasi Nilai Belanja (Total Spent) Transaksi Kafe
dengan Mempertimbangkan Faktor Inflasi.
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "cafe_sales_clean.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

st.set_page_config(
    page_title="Prediksi Belanja Kafe | Capstone ML",
    page_icon="☕",
    layout="wide",
)


# ----------------------------------------------------------------------
# Loaders (cache supaya tidak reload berulang)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))


@st.cache_data
def load_metadata():
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_comparison():
    return pd.read_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))


@st.cache_data
def load_importance():
    return pd.read_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"))


@st.cache_data
def load_residuals():
    return pd.read_csv(os.path.join(REPORTS_DIR, "residuals_test.csv"))


df = load_data()
model = load_model()
metadata = load_metadata()
comparison_df = load_comparison()
importance_df = load_importance()
residual_df = load_residuals()

# ----------------------------------------------------------------------
# Sidebar navigasi
# ----------------------------------------------------------------------
st.sidebar.title("☕ Capstone Project ML")
st.sidebar.caption("Prediksi Nilai Belanja Kafe dengan Faktor Inflasi")
page = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Ringkasan Proyek",
        "📊 Dashboard EDA",
        "🤖 Model Demo",
        "📈 Evaluasi Model",
        "🔍 Interpretasi Hasil",
        "📄 Dokumentasi",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model terbaik:** {metadata['best_model']}")
st.sidebar.markdown(f"**Jumlah data (bersih):** {len(df):,} transaksi")

ITEMS = sorted(df["Item"].unique())
PAYMENTS = sorted(df["Payment Method"].unique())
LOCATIONS = sorted(df["Location"].unique())

# =========================================================================
# HALAMAN 1: RINGKASAN PROYEK
# =========================================================================
if page == "🏠 Ringkasan Proyek":
    st.title("☕ Prediksi Estimasi Nilai Belanja Kafe (dengan Faktor Inflasi)")
    st.markdown(
        """
        ### Problem Statement
        Sebuah kafe ingin memperkirakan **estimasi nilai belanja (Total Spent)** dari
        sebuah transaksi hanya berdasarkan **pilihan menu dan konteks transaksi**
        (waktu, lokasi, metode pembayaran) — **tanpa mengetahui quantity/harga
        final di muka** — sekaligus mempertimbangkan kondisi **inflasi bulanan**
        sebagai faktor makroekonomi eksternal.

        Estimasi ini berguna untuk **forecasting pendapatan** dan **perencanaan
        operasional** (stok, staffing) secara real-time saat pelanggan baru mulai
        memilih menu.

        ### Sumber Data
        - **`dirty_cafe_sales.csv`** (Kaggle) — data transaksi kafe (sengaja "kotor":
          mengandung placeholder `ERROR`/`UNKNOWN`, missing values, dan duplikat).
        - **`Data_Inflasi.xlsx`** (Bank Indonesia) — data inflasi bulanan tahun 2023.

        ### Metrik Kesuksesan
        - Model regresi dengan **RMSE serendah mungkin** pada data test, sekaligus
          **dapat diinterpretasikan** (business-friendly) oleh stakeholder non-teknis.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transaksi (bersih)", f"{len(df):,}")
    col2.metric("Jumlah Fitur", f"{len(metadata['numeric_features']) + len(metadata['categorical_features'])}")
    col3.metric("Model Terbaik", metadata["best_model"])
    best_res = metadata["results"][metadata["best_model"]]["test"]
    col4.metric("R² (Test)", f"{best_res['R2']:.3f}")

    st.markdown("### Statistik Deskriptif Data")
    st.dataframe(df.describe().T, use_container_width=True)

# =========================================================================
# HALAMAN 2: DASHBOARD EDA
# =========================================================================
elif page == "📊 Dashboard EDA":
    st.title("📊 Dashboard EDA (Interaktif)")

    tab1, tab2, tab3 = st.tabs(["Distribusi & Item", "Waktu & Inflasi", "Korelasi"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Total Spent", nbins=30, title="Distribusi Total Spent")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            item_counts = df["Item"].value_counts().reset_index()
            item_counts.columns = ["Item", "Jumlah Transaksi"]
            fig = px.bar(item_counts, x="Item", y="Jumlah Transaksi", title="Frekuensi Penjualan per Item")
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x="Item", y="Total Spent", title="Sebaran Total Spent per Item", color="Item")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        monthly = df.groupby(["Year", "Month"]).agg(
            Total_Penjualan=("Total Spent", "sum"),
            Inflasi=("Inflasi", "mean"),
        ).reset_index()
        monthly["Periode"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)

        fig = px.line(monthly, x="Periode", y="Total_Penjualan", markers=True,
                      title="Tren Total Penjualan Bulanan")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(monthly, x="Periode", y="Inflasi", markers=True,
                       title="Tren Inflasi Bulanan (%)")
        fig2.update_traces(line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

        payment_dist = df["Payment Method"].value_counts().reset_index()
        payment_dist.columns = ["Metode Bayar", "Jumlah"]
        fig3 = px.pie(payment_dist, names="Metode Bayar", values="Jumlah", title="Distribusi Metode Pembayaran")
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        numeric_cols = ["Total Spent", "Quantity", "Price Per Unit", "Inflasi", "Month", "DayOfWeek"]
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         title="Correlation Heatmap Fitur Numerik")
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "💡 **Insight:** `Total Spent` berkorelasi kuat dengan `Quantity` & `Price Per Unit` "
            "(hubungan matematis langsung), tetapi korelasinya dengan `Inflasi` sangat lemah — "
            "mengindikasikan penjualan kafe relatif *resilient* terhadap gejolak makroekonomi."
        )

# =========================================================================
# HALAMAN 3: MODEL DEMO
# =========================================================================
elif page == "🤖 Model Demo":
    st.title("🤖 Model Demo — Coba Prediksi Sendiri")
    st.markdown(
        "Masukkan detail transaksi berikut untuk mendapatkan **estimasi nilai belanja** "
        f"pelanggan menggunakan model **{metadata['best_model']}**."
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            item = st.selectbox("Item / Menu", ITEMS)
            payment = st.selectbox("Metode Pembayaran", PAYMENTS)
        with c2:
            location = st.selectbox("Lokasi", LOCATIONS)
            month = st.slider("Bulan", 1, 12, 6)
        with c3:
            day_of_week = st.selectbox(
                "Hari", options=list(range(7)),
                format_func=lambda x: ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][x],
            )
            inflasi = st.slider("Tingkat Inflasi (%)", float(df["Inflasi"].min()), float(df["Inflasi"].max()), float(df["Inflasi"].median()))

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

    if submitted:
        is_weekend = 1 if day_of_week in [5, 6] else 0
        input_df = pd.DataFrame([{
            "Inflasi": inflasi,
            "Year": 2023,
            "Month": month,
            "DayOfWeek": day_of_week,
            "IsWeekend": is_weekend,
            "Item": item,
            "Payment Method": payment,
            "Location": location,
        }])

        pred = model.predict(input_df)[0]
        st.success(f"### 💰 Estimasi Nilai Belanja: **${pred:,.2f}**")

        item_avg = df[df["Item"] == item]["Total Spent"].mean()
        st.caption(f"Rata-rata historis untuk item **{item}**: ${item_avg:,.2f}")

        fig = px.bar(
            x=["Prediksi Model", f"Rata-rata Historis ({item})"],
            y=[pred, item_avg],
            title="Perbandingan Prediksi vs Rata-rata Historis",
            color=["Prediksi Model", f"Rata-rata Historis ({item})"],
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# HALAMAN 4: EVALUASI MODEL
# =========================================================================
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model")

    st.markdown("### Tabel Perbandingan Performa Semua Model")
    st.dataframe(comparison_df.style.highlight_min(subset=[c for c in comparison_df.columns if "RMSE" in c or "MAE" in c], color="#bbf7d0")
                 .highlight_max(subset=[c for c in comparison_df.columns if "R2" in c], color="#bbf7d0"),
                 use_container_width=True)

    st.markdown(f"### Model Terpilih: **{metadata['best_model']}**")
    best_test = metadata["results"][metadata["best_model"]]["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{best_test['R2']:.3f}")
    c2.metric("MAE", f"{best_test['MAE']:.3f}")
    c3.metric("MSE", f"{best_test['MSE']:.3f}")
    c4.metric("RMSE", f"{best_test['RMSE']:.3f}")

    st.markdown("### Visualisasi Evaluasi (Test Set)")
    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(FIG_DIR, "actual_vs_predicted.png"), caption="Actual vs Predicted", use_container_width=True)
    with c2:
        st.image(os.path.join(FIG_DIR, "residual_plot.png"), caption="Residual Plot", use_container_width=True)

    st.image(os.path.join(FIG_DIR, "residual_distribution.png"), caption="Distribusi Residual", use_container_width=True)

    with st.expander("Lihat detail data residual (test set)"):
        st.dataframe(residual_df, use_container_width=True)

# =========================================================================
# HALAMAN 5: INTERPRETASI HASIL
# =========================================================================
elif page == "🔍 Interpretasi Hasil":
    st.title("🔍 Interpretasi Hasil & Insights Bisnis")

    st.markdown("### Feature Importance (Permutation Importance)")
    st.image(os.path.join(FIG_DIR, "feature_importance.png"), use_container_width=True)
    st.dataframe(importance_df, use_container_width=True)

    shap_path = os.path.join(FIG_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        st.markdown("### SHAP Summary Plot (Random Forest)")
        st.image(shap_path, use_container_width=True)

    st.markdown(
        """
        ### Insight Bisnis Utama
        1. **Pilihan menu (`Item`) adalah pendorong utama** nilai belanja pelanggan —
           jauh lebih berpengaruh daripada faktor lain.
        2. **Faktor waktu** (bulan, hari) memberi pengaruh kecil namun konsisten,
           berguna untuk perencanaan staffing/promosi musiman.
        3. **Inflasi bulanan bukan prediktor kuat** — mendukung temuan bahwa bisnis
           kafe relatif *resilient* terhadap gejolak ekonomi makro, sehingga strategi
           bisnis lebih baik difokuskan pada bauran menu & pengalaman pelanggan
           daripada mitigasi risiko makroekonomi jangka pendek.

        ### Rekomendasi
        - Fokuskan strategi *upselling* pada kombinasi menu dengan nilai belanja tinggi.
        - Gunakan estimasi model ini untuk **forecasting pendapatan harian/mingguan**
          tanpa perlu menunggu transaksi selesai.
        """
    )

# =========================================================================
# HALAMAN 6: DOKUMENTASI
# =========================================================================
elif page == "📄 Dokumentasi":
    st.title("📄 Dokumentasi Proyek")

    st.markdown(
        f"""
        ### Dataset
        - **dirty_cafe_sales.csv** — 10.000 transaksi mentah dari Kaggle (sengaja berisi
          data kotor: `ERROR`, `UNKNOWN`, missing value, duplikat).
        - **Data_Inflasi.xlsx** — data inflasi bulanan Bank Indonesia tahun 2023.
        - Setelah preprocessing: **{len(df):,} transaksi bersih**.

        ### Metodologi (Pipeline)
        1. **Data Acquisition** — load kedua sumber data.
        2. **Cleaning** — handle placeholder, missing values, duplikat, outlier (IQR).
        3. **Feature Engineering** — join inflasi bulanan, ekstraksi fitur waktu
           (`Year`, `Month`, `DayOfWeek`, `IsWeekend`).
        4. **Modeling** — Linear Regression & Random Forest (tuning via GridSearchCV).
        5. **Evaluasi** — R², MAE, MSE, RMSE, residual analysis.
        6. **Interpretasi** — Permutation Importance & SHAP.
        7. **Deployment** — aplikasi Streamlit ini.

        ### Fitur yang Digunakan Model
        - **Numerik:** {', '.join(metadata['numeric_features'])}
        - **Kategorikal:** {', '.join(metadata['categorical_features'])}
        - **Target:** {metadata['target']}

        > ⚠️ **Catatan penting:** `Quantity` dan `Price Per Unit` sengaja **tidak**
        > dipakai sebagai fitur karena `Total Spent = Quantity x Price Per Unit`
        > secara matematis persis — jika dipakai, model menjadi trivial (data leakage).

        ### Cara Menjalankan Aplikasi Secara Lokal
        ```bash
        git clone <repo-url>
        cd capstone-project-data-mining
        pip install -r requirements.txt
        python src/data_preprocessing.py   # generate data bersih
        python src/train_model.py          # training & tuning model
        python src/evaluate_model.py       # generate visualisasi evaluasi
        streamlit run app/app.py
        ```

        ### Struktur Repository
        Lihat `README.md` pada root repository.
        """
    )
