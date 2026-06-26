import os
import pandas as pd
import numpy as np
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_CSV = os.path.join(DATA_DIR, 'raw_superstore.csv')
CLEANED_CSV = os.path.join(DATA_DIR, 'cleaned_superstore.csv')

def load_raw_data():
    print('=' * 60)
    print('  STEP 2: DATA CLEANING')
    print('=' * 60)
    df = pd.read_csv(RAW_CSV, encoding='windows-1252')
    print(f'\n[INFO] Loaded raw data: {len(df):,} rows, {df.shape[1]} columns')
    return df

def clean_data(df):
    initial_rows = len(df)
    cleaning_log = []
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    cleaning_log.append(('Duplicate rows', dup_count))
    print(f'\n[CLEAN] Removed {dup_count:,} duplicate rows')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y', errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y', errors='coerce')
    print(f'[CLEAN] Converted dates to datetime objects')
    missing_dates = df['Order Date'].isna().sum()
    if missing_dates > 0:
        df = df.dropna(subset=['Order Date'])
        cleaning_log.append(('Missing Order Date', missing_dates))
        print(f'[CLEAN] Removed {missing_dates:,} rows with invalid Order Date')
    invalid_sales = (df['Sales'] < 0).sum()
    if invalid_sales > 0:
        df = df[df['Sales'] >= 0]
        cleaning_log.append(('Negative Sales', invalid_sales))
        print(f'[CLEAN] Removed {invalid_sales:,} rows with Sales < 0')
    invalid_qty = (df['Quantity'] <= 0).sum()
    if invalid_qty > 0:
        df = df[df['Quantity'] > 0]
        cleaning_log.append(('Non-positive Quantity', invalid_qty))
        print(f'[CLEAN] Removed {invalid_qty:,} rows with Quantity <= 0')
    final_rows = len(df)
    total_removed = initial_rows - final_rows
    print(f"\n{'─' * 60}")
    print(f'  CLEANING SUMMARY')
    print(f"{'─' * 60}")
    print(f"  {'Reason':<45} {'Rows Removed':>12}")
    print(f"  {'─' * 57}")
    for reason, count in cleaning_log:
        print(f'  {reason:<45} {count:>12,}')
    if not cleaning_log:
        print(f"  {'No data removed (clean dataset)':<45} {0:>12,}")
    print(f"  {'─' * 57}")
    print(f"  {'TOTAL REMOVED':<45} {total_removed:>12,}")
    print(f"  {'REMAINING ROWS':<45} {final_rows:>12,}")
    print(f'  Retention rate: {final_rows / initial_rows * 100:.1f}%')
    return df

def validate_cleaned_data(df):
    print(f"\n{'─' * 60}")
    print(f'  VALIDATION CHECKS')
    print(f"{'─' * 60}")
    checks = {'Order Date is datetime': pd.api.types.is_datetime64_any_dtype(df['Order Date']), 'All Sales >= 0': (df['Sales'] >= 0).all(), 'All Quantity > 0': (df['Quantity'] > 0).all()}
    all_passed = True
    for check_name, passed in checks.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'  {status}  {check_name}')
        if not passed:
            all_passed = False
    if all_passed:
        print('\n  [SUCCESS] All validation checks passed!')
    else:
        print('\n  [WARNING] Some checks failed — review the data.')
    return all_passed

def save_cleaned_data(df):
    print(f'\n[INFO] Saving cleaned data to {CLEANED_CSV}...')
    df.to_csv(CLEANED_CSV, index=False)
    file_size_mb = os.path.getsize(CLEANED_CSV) / (1024 * 1024)
    print(f'[SUCCESS] Saved ({file_size_mb:.1f} MB)')
if __name__ == '__main__':
    df = load_raw_data()
    df = clean_data(df)
    validate_cleaned_data(df)
    save_cleaned_data(df)
    print('\n' + '=' * 60)
    print('  STEP 2 COMPLETE ✓')
    print('=' * 60)
