"""
utils.py
---------
Fungsi utilitas yang dipakai bersama oleh notebooks, src/, dan app/ Streamlit.
"""

import os
import json
import joblib
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROCESSED_DIR, "cafe_sales_clean.csv"))


def load_best_model():
    return joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))


def load_metadata() -> dict:
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        return json.load(f)


def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"))


def load_residuals() -> pd.DataFrame:
    return pd.read_csv(os.path.join(REPORTS_DIR, "residuals_test.csv"))
