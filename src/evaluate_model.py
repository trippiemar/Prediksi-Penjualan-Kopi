"""
evaluate_model.py
-------------------
Membuat visualisasi evaluasi untuk model terbaik:
  - Actual vs Predicted scatter
  - Residual plot & distribusi residual
  - Feature importance (permutation importance)
  - Tabel perbandingan performa semua model
Semua gambar disimpan di reports/figures/ (dipakai juga oleh aplikasi Streamlit).
"""

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

sns.set_style("whitegrid")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    residual_df = pd.read_csv(os.path.join(REPORTS_DIR, "residuals_test.csv"))
    importance_df = pd.read_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"))
    metadata = json.load(open(os.path.join(MODELS_DIR, "metadata.json")))

    # 1. Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(residual_df["y_true"], residual_df["y_pred"], alpha=0.4, s=15, color="#2563eb")
    lims = [residual_df["y_true"].min(), residual_df["y_true"].max()]
    plt.plot(lims, lims, "r--", linewidth=2, label="Garis Ideal (y=x)")
    plt.xlabel("Nilai Aktual (Total Spent)")
    plt.ylabel("Nilai Prediksi")
    plt.title(f"Actual vs Predicted - {metadata['best_model']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "actual_vs_predicted.png"), dpi=120)
    plt.close()

    # 2. Residual plot
    plt.figure(figsize=(7, 5))
    plt.scatter(residual_df["y_pred"], residual_df["residual"], alpha=0.4, s=15, color="#059669")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Nilai Prediksi")
    plt.ylabel("Residual (Aktual - Prediksi)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "residual_plot.png"), dpi=120)
    plt.close()

    # 3. Distribusi residual
    plt.figure(figsize=(7, 5))
    sns.histplot(residual_df["residual"], kde=True, color="#7c3aed")
    plt.xlabel("Residual")
    plt.title("Distribusi Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "residual_distribution.png"), dpi=120)
    plt.close()

    # 4. Feature importance
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"], importance_df["importance_mean"], color="#f59e0b")
    plt.xlabel("Permutation Importance (penurunan skor R2)")
    plt.title("Feature Importance - Model Terbaik")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=120)
    plt.close()

    # 5. Tabel perbandingan performa
    rows = []
    for model_name, res in metadata["results"].items():
        rows.append({
            "Model": model_name,
            "R2 (Val)": res["val"]["R2"],
            "MAE (Val)": res["val"]["MAE"],
            "RMSE (Val)": res["val"]["RMSE"],
            "R2 (Test)": res["test"]["R2"],
            "MAE (Test)": res["test"]["MAE"],
            "RMSE (Test)": res["test"]["RMSE"],
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print(comparison_df.to_string(index=False))
    print(f"\nSemua figure evaluasi tersimpan di: {FIG_DIR}")


if __name__ == "__main__":
    main()
