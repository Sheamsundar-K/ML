import os
import shutil
import pandas as pd
import numpy as np
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_CSV = os.path.join(DATA_DIR, 'raw_superstore.csv')
os.makedirs(DATA_DIR, exist_ok=True)

def load_dataset():
    print('=' * 60)
    print('  STEP 1: DATA LOADING & INITIAL EXPLORATION')
    print('=' * 60)
    import kagglehub
    print('\n[INFO] Fetching Superstore dataset from Kaggle...')
    path = kagglehub.dataset_download('vivek468/superstore-dataset-final')
    csv_file = None
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith('.csv'):
                csv_file = os.path.join(root, name)
                break
    if not csv_file:
        raise FileNotFoundError('Could not find the CSV file in the downloaded dataset.')
    print(f'[INFO] Found dataset at: {csv_file}')
    print(f'[INFO] Copying to {RAW_CSV}...')
    shutil.copy(csv_file, RAW_CSV)
    df = pd.read_csv(RAW_CSV, encoding='windows-1252')
    print(f'[SUCCESS] Dataset loaded — {len(df):,} rows')
    return df

def explore_dataset(df):
    print('\n' + '─' * 60)
    print('  DATASET SHAPE')
    print('─' * 60)
    print(f'  Rows:    {df.shape[0]:,}')
    print(f'  Columns: {df.shape[1]}')
    print('\n' + '─' * 60)
    print('  COLUMN INFORMATION')
    print('─' * 60)
    print(f"\n{'Column':<20} {'Dtype':<15} {'Non-Null':>10} {'Null':>8} {'Null%':>8}")
    print('─' * 65)
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        print(f'  {col:<18} {str(df[col].dtype):<15} {non_null:>10,} {null_count:>8,} {null_pct:>7.2f}%')
    print('\n' + '─' * 60)
    print('  FIRST 5 ROWS')
    print('─' * 60)
    print(df.head().to_string())
    print('\n' + '─' * 60)
    print('  NUMERIC SUMMARY STATISTICS')
    print('─' * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(df[numeric_cols].describe().to_string())
    print('\n' + '─' * 60)
    print('  UNIQUE VALUES PER COLUMN')
    print('─' * 60)
    for col in df.columns:
        print(f'  {col:<20}: {df[col].nunique():>8,} unique values')
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y', errors='coerce')
        print(f"\n[INFO] Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
    if 'Category' in df.columns:
        print(f'\n[INFO] Category distribution:')
        cats = df['Category'].value_counts()
        for cat, count in cats.items():
            print(f'  {cat:<25}: {count:>8,}')
if __name__ == '__main__':
    df = load_dataset()
    explore_dataset(df)
    print('\n' + '=' * 60)
    print('  STEP 1 COMPLETE ✓')
    print('=' * 60)
