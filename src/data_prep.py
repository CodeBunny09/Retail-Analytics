# import pandas as pd
# import numpy as np

# def load_datasets():
#     date_cols = ['Date']
#     df_features = pd.read_csv("../data/raw/Features.csv", parse_dates=date_cols)
#     df_sales = pd.read_csv("../data/raw/sales.csv", parse_dates=date_cols)
#     df_stores = pd.read_csv("../data/raw/stores.csv")

#     # Ensure 'Date' columns are datetime
#     df_features['Date'] = pd.to_datetime(df_features['Date'], dayfirst=True, errors='coerce')
#     df_sales['Date'] = pd.to_datetime(df_sales['Date'], dayfirst=True, errors='coerce')
    
#     return df_features, df_sales, df_stores

# def drop_sparse_columns(df, threshold=0.3):
#     limit = int((1 - threshold) * len(df))
#     return df.dropna(thresh=limit, axis=1)

# def merge_datasets(df_sales, df_stores, df_features):
#     df = pd.merge(df_sales, df_stores, on='Store', how='left')
#     df = pd.merge(df, df_features, on=['Store', 'Date'], how='left')
#     return df

# def handle_missing_values(df):
#     # Fill numeric columns with median
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
#     # Fill non-numeric with 'Unknown'
#     df = df.fillna('Unknown')
#     return df

# def initial_eda(df):
#     print(f'Shape: {df.shape}\n\n')
#     print(f'Data types:\n{df.dtypes}\n\n')
#     print(f'Description:\n{df.describe()}\n\n')
#     print(f'Info:\n{df.info()}\n\n')

# def save_cleaned_data(df, path):
#     df.to_csv(path, index=False)
#     print(f"Cleaned data saved to {path}")


"""Data loading, cleaning, merging, and basic EDA."""
import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_logger

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "Date"

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_csv(name: str, parse_dates=True):
    path = RAW_DIR / name
    df = pd.read_csv(path)
    if parse_dates and DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    logger.info(f"Loaded {name} -> {df.shape}")
    return df


def load_datasets():
    """Load all three raw CSVs and return features, sales, stores."""
    df_features = load_csv("Features.csv")
    df_sales = load_csv("sales.csv")
    df_stores = load_csv("stores.csv", parse_dates=False)

    
    # Ensure 'Date' columns are datetime
    df_features['Date'] = pd.to_datetime(df_features['Date'], dayfirst=True, errors='coerce')
    df_sales['Date'] = pd.to_datetime(df_sales['Date'], dayfirst=True, errors='coerce')

    return df_features, df_sales, df_stores

# ---------------------------------------------------------------------------
# Cleaning & merging
# ---------------------------------------------------------------------------

def drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.5, *, keep_cols=None):
    keep_cols = keep_cols or []
    limit = int((1 - threshold) * len(df))
    to_drop = [c for c in df.columns if c not in keep_cols and df[c].isna().sum() > limit]
    logger.info(f"Dropping sparse columns: {to_drop}")
    return df.drop(columns=to_drop)


def merge_datasets(df_sales, df_stores, df_features):
    logger.info("Merging datasets …")
    df = pd.merge(df_sales, df_stores, on="Store", how="left")
    df = pd.merge(df, df_features, on=["Store", DATE_COL], how="left")
    logger.info(f"Merged dataset shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.fillna("Unknown")
    return df


def initial_eda(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nDescription:\n", df.describe(include="all"))


def save_cleaned_data(df: pd.DataFrame, name: str = "cleaned_data.csv"):
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    logger.info(f"Saved cleaned data -> {out}")
