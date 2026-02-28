"""
src/data_cleaning.py
--------------------
Handles all data cleaning operations:
  - Loading raw CSV
  - Removing duplicates
  - Imputing missing values (group-wise median)
  - Detecting & treating outliers (IQR method)
  - Enforcing correct data types
"""

import numpy as np
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────
RAW_PATH = "data/life_expectancy.csv"
CLEAN_PATH = "data/life_expectancy_clean.csv"

NUMERIC_COLS = [
    "Life expectancy",
    "Adult Mortality",
    "infant deaths",
    "under-five deaths",
    "GDP",
    "Schooling",
    "BMI",
    "Alcohol",
    "percentage expenditure",
    "Population",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV and enforce basic types."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Year"] = df["Year"].astype(int)
    df["Status"] = df["Status"].astype("category")
    print(f"[load]  Shape: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[dedup] Removed {before - len(df)} duplicate rows → {len(df)} rows remain")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using group-wise median (Country + Status).
    Falls back to global median if a group has no non-null values.
    """
    missing_before = df[NUMERIC_COLS].isnull().sum().sum()

    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            # Group-wise median per Status
            group_medians = df.groupby("Status")[col].transform("median")
            global_median = df[col].median()
            df[col] = df[col].fillna(group_medians).fillna(global_median)

    missing_after = df[NUMERIC_COLS].isnull().sum().sum()
    print(f"[impute] Missing values: {missing_before} → {missing_after}")
    return df


def treat_outliers_iqr(df: pd.DataFrame, cols: list = None, factor: float = 3.0) -> pd.DataFrame:
    """
    Cap outliers at [Q1 - factor*IQR, Q3 + factor*IQR] (Winsorization).
    A factor of 3.0 is conservative — only extreme values are capped.
    """
    cols = cols or NUMERIC_COLS
    outlier_counts = {}

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_counts[col] = n_out
        df[col] = df[col].clip(lower=lower, upper=upper)

    total = sum(outlier_counts.values())
    print(f"[outliers] Capped {total} outlier values across {len(cols)} columns")
    return df


def clean_pipeline(path: str = RAW_PATH, save: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Parameters
    ----------
    path : str
        Path to raw CSV file.
    save : bool
        If True, saves the cleaned DataFrame to CLEAN_PATH.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    print("=" * 50)
    print("DATA CLEANING PIPELINE")
    print("=" * 50)

    df = load_raw(path)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = treat_outliers_iqr(df)

    # Final type checks
    df["Year"] = df["Year"].astype(int)
    df["Country"] = df["Country"].astype(str)

    if save:
        df.to_csv(CLEAN_PATH, index=False)
        print(f"[save]  Clean data saved to '{CLEAN_PATH}'")

    print(f"\nFinal shape: {df.shape}")
    print("=" * 50)
    return df


if __name__ == "__main__":
    clean_pipeline()
