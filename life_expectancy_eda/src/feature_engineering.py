"""
src/feature_engineering.py
---------------------------
Feature Engineering pipeline:
  - Label encode categorical columns
  - Standardise numeric features (StandardScaler)
  - Drop low-variance features (VarianceThreshold)
  - Create derived features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold


# ─── Derived Features ─────────────────────────────────────────────────────────

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-relevant features.

    New columns
    -----------
    log_GDP          : log(GDP + 1) — reduces right skew
    GDP_per_schooling: GDP / Schooling — proxy for economic efficiency
    mortality_ratio  : Adult Mortality / (infant deaths + 1)
    """
    df = df.copy()
    df["log_GDP"] = np.log1p(df["GDP"])
    df["GDP_per_schooling"] = df["GDP"] / (df["Schooling"] + 1e-6)
    df["mortality_ratio"] = df["Adult Mortality"] / (df["infant deaths"] + 1)
    print("[feat_eng] Derived features created: log_GDP, GDP_per_schooling, mortality_ratio")
    return df


# ─── Encoding ─────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode 'Status' (Developed=1 / Developing=0)."""
    df = df.copy()
    le = LabelEncoder()
    df["Status_enc"] = le.fit_transform(df["Status"].astype(str))
    print(f"[feat_eng] Encoded 'Status': classes → {list(le.classes_)}")
    return df


# ─── Scaling ──────────────────────────────────────────────────────────────────

SCALE_COLS = [
    "Life expectancy",
    "Adult Mortality",
    "infant deaths",
    "Under-five deaths",
    "GDP",
    "Schooling",
    "BMI",
    "Alcohol",
    "percentage expenditure",
    "Population",
    "log_GDP",
    "GDP_per_schooling",
    "mortality_ratio",
]


def standardise_features(df: pd.DataFrame, cols: list = None) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScale numeric columns.

    Returns
    -------
    df_scaled  : DataFrame with scaled columns replacing originals
    scaler     : Fitted StandardScaler (for inverse transform later)
    """
    cols = cols or [c for c in SCALE_COLS if c in df.columns]
    df = df.copy()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    print(f"[feat_eng] Standardised {len(cols)} numeric features")
    return df, scaler


# ─── Variance Filtering ───────────────────────────────────────────────────────

def drop_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Drop columns whose variance (after scaling) is below `threshold`.
    Categorical and non-numeric columns are preserved.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)

    low_var = [col for col, keep in zip(numeric_df.columns, selector.get_support()) if not keep]
    if low_var:
        df = df.drop(columns=low_var)
        print(f"[feat_eng] Dropped low-variance columns: {low_var}")
    else:
        print("[feat_eng] No low-variance columns found")
    return df


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def feature_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    End-to-end feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame  — cleaned DataFrame from data_cleaning.py

    Returns
    -------
    df_ready : engineered + scaled DataFrame
    scaler   : fitted StandardScaler
    """
    print("=" * 50)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 50)

    df = create_derived_features(df)
    df = encode_categoricals(df)
    df, scaler = standardise_features(df)
    df = drop_low_variance(df)

    print(f"\nFinal feature shape: {df.shape}")
    print("=" * 50)
    return df, scaler


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_cleaning import clean_pipeline

    df_clean = clean_pipeline(save=False)
    df_ready, scaler = feature_pipeline(df_clean)
    print(df_ready.head())
