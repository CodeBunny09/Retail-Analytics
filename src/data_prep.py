import pandas as pd
import numpy as np

def load_datasets():
    date_cols = ['Date']
    df_features = pd.read_csv("../data/raw/Features.csv", parse_dates=date_cols)
    df_sales = pd.read_csv("../data/raw/sales.csv", parse_dates=date_cols)
    df_stores = pd.read_csv("../data/raw/stores.csv")

    # Ensure 'Date' columns are datetime
    df_features['Date'] = pd.to_datetime(df_features['Date'], dayfirst=True, errors='coerce')
    df_sales['Date'] = pd.to_datetime(df_sales['Date'], dayfirst=True, errors='coerce')
    
    return df_features, df_sales, df_stores

def drop_sparse_columns(df, threshold=0.3):
    limit = int((1 - threshold) * len(df))
    return df.dropna(thresh=limit, axis=1)

def merge_datasets(df_sales, df_stores, df_features):
    df = pd.merge(df_sales, df_stores, on='Store', how='left')
    df = pd.merge(df, df_features, on=['Store', 'Date'], how='left')
    return df

def handle_missing_values(df):
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Fill non-numeric with 'Unknown'
    df = df.fillna('Unknown')
    return df

def initial_eda(df):
    print(f'Shape: {df.shape}\n\n')
    print(f'Data types:\n{df.dtypes}\n\n')
    print(f'Description:\n{df.describe()}\n\n')
    print(f'Info:\n{df.info()}\n\n')

def save_cleaned_data(df, path):
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to {path}")
