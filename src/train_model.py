"""
train_model.py
---------------
Melatih dan men-tuning 2 model regresi untuk memprediksi `Total Spent`:
  1. Linear Regression (baseline, model interpretable)
  2. Random Forest Regressor (model ensemble, non-linear)

Tuning hyperparameter Random Forest menggunakan GridSearchCV.
Evaluasi menggunakan R2, MAE, MSE, RMSE + residual analysis.
Model terbaik (berdasarkan RMSE test set) disimpan sebagai models/best_model.pkl
beserta pipeline preprocessing-nya (models/preprocessing.pkl, disatukan dalam 1
sklearn Pipeline agar mudah dipakai saat deployment).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cafe_sales_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# CATATAN PENTING (menghindari data leakage):
# Total Spent = Quantity x Price Per Unit secara matematis PERSIS.
# Jika keduanya dipakai sebagai fitur, model hanya belajar perkalian (R2=1.0,
# trivial, bukan pembelajaran yang bermakna). Karena itu Quantity & Price Per
# Unit TIDAK diikutkan sebagai fitur. Problem yang lebih realistis & berguna
# secara bisnis: memperkirakan estimasi nilai belanja (expected spend) seorang
# pelanggan HANYA dari pilihan menu & konteks transaksi (waktu, lokasi, metode
# bayar, kondisi inflasi) -- berguna untuk forecasting revenue sebelum checkout
# selesai / quantity final diketahui.
NUMERIC_FEATURES = ["Inflasi", "Year", "Month", "DayOfWeek", "IsWeekend"]
CATEGORICAL_FEATURES = ["Item", "Payment Method", "Location"]
TARGET = "Total Spent"


def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def evaluate(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Train - validation - test split (70-15-15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    results = {}
    fitted_pipelines = {}

    # ----------------- Model 1: Linear Regression -----------------
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    fitted_pipelines["Linear Regression"] = lr_pipeline
    results["Linear Regression"] = {
        "val": evaluate(y_val, lr_pipeline.predict(X_val)),
        "test": evaluate(y_test, lr_pipeline.predict(X_test)),
        "best_params": {},
    }

    # ----------------- Model 2: Random Forest + GridSearchCV -----------------
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(random_state=42)),
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    fitted_pipelines["Random Forest"] = best_rf
    results["Random Forest"] = {
        "val": evaluate(y_val, best_rf.predict(X_val)),
        "test": evaluate(y_test, best_rf.predict(X_test)),
        "best_params": grid_search.best_params_,
    }

    # ----------------- Pilih model terbaik berdasarkan RMSE test -----------------
    best_model_name = min(results, key=lambda k: results[k]["test"]["RMSE"])
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\nModel terbaik: {best_model_name}")
    print(json.dumps(results[best_model_name], indent=2))

    # ----------------- Feature importance (permutation importance) -----------------
    perm = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # ----------------- Residuals untuk analisis -----------------
    y_test_pred = best_pipeline.predict(X_test)
    residual_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_test_pred,
        "residual": y_test.values - y_test_pred,
    })

    # ----------------- Simpan semua artefak -----------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(fitted_pipelines["Linear Regression"], os.path.join(MODELS_DIR, "linear_regression_model.pkl"))
    joblib.dump(fitted_pipelines["Random Forest"], os.path.join(MODELS_DIR, "random_forest_model.pkl"))

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump({
            "best_model": best_model_name,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "results": results,
        }, f, indent=2)

    importance_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False)
    residual_df.to_csv(os.path.join(REPORTS_DIR, "residuals_test.csv"), index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(os.path.join(BASE_DIR, "data", "processed", "test_set.csv"), index=False)

    print("\nSemua artefak model & laporan berhasil disimpan.")


if __name__ == "__main__":
    main()
