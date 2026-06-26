import os
import pandas as pd
import numpy as np
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CLEANED_CSV = os.path.join(DATA_DIR, 'cleaned_superstore.csv')
DAILY_CSV = os.path.join(DATA_DIR, 'daily_sales.csv')
WEEKLY_CSV = os.path.join(DATA_DIR, 'weekly_sales.csv')
MONTHLY_CSV = os.path.join(DATA_DIR, 'monthly_sales.csv')

def load_cleaned_data():
    print('=' * 60)
    print('  STEP 3: FEATURE ENGINEERING')
    print('=' * 60)
    df = pd.read_csv(CLEANED_CSV, parse_dates=['Order Date', 'Ship Date'])
    print(f'\n[INFO] Loaded cleaned data: {len(df):,} rows')
    return df

def create_daily_aggregation(df):
    print('\n[INFO] Creating daily aggregation...')
    df['Date'] = df['Order Date'].dt.date
    daily = df.groupby('Date').agg(Sales=('Sales', 'sum'), Profit=('Profit', 'sum'), TotalQuantity=('Quantity', 'sum'), NumOrders=('Order ID', 'nunique'), NumCustomers=('Customer ID', 'nunique'), NumProducts=('Product ID', 'nunique'), AvgOrderValue=('Sales', 'mean')).reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date').reset_index(drop=True)
    full_date_range = pd.date_range(start=daily['Date'].min(), end=daily['Date'].max(), freq='D')
    daily = daily.set_index('Date').reindex(full_date_range, fill_value=0).reset_index()
    daily.rename(columns={'index': 'Date'}, inplace=True)
    print(f"  → {len(daily)} daily records ({daily['Date'].min().date()} to {daily['Date'].max().date()})")
    return daily

def create_weekly_aggregation(df):
    print('[INFO] Creating weekly aggregation...')
    df['Week'] = df['Order Date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = df.groupby('Week').agg(Sales=('Sales', 'sum'), Profit=('Profit', 'sum'), TotalQuantity=('Quantity', 'sum'), NumOrders=('Order ID', 'nunique'), NumCustomers=('Customer ID', 'nunique')).reset_index()
    weekly = weekly.sort_values('Week').reset_index(drop=True)
    print(f'  → {len(weekly)} weekly records')
    return weekly

def create_monthly_aggregation(df):
    print('[INFO] Creating monthly aggregation...')
    df['Month'] = df['Order Date'].dt.to_period('M').apply(lambda r: r.start_time)
    monthly = df.groupby('Month').agg(Sales=('Sales', 'sum'), Profit=('Profit', 'sum'), TotalQuantity=('Quantity', 'sum'), NumOrders=('Order ID', 'nunique'), NumCustomers=('Customer ID', 'nunique'), NumProducts=('Product ID', 'nunique')).reset_index()
    monthly = monthly.sort_values('Month').reset_index(drop=True)
    print(f'  → {len(monthly)} monthly records')
    return monthly

def add_time_features(daily):
    print('\n[INFO] Adding time features...')
    daily['Year'] = daily['Date'].dt.year
    daily['Month'] = daily['Date'].dt.month
    daily['DayOfWeek'] = daily['Date'].dt.dayofweek
    daily['DayOfMonth'] = daily['Date'].dt.day
    daily['WeekOfYear'] = daily['Date'].dt.isocalendar().week.astype(int)
    daily['Quarter'] = daily['Date'].dt.quarter
    daily['IsWeekend'] = (daily['DayOfWeek'] >= 5).astype(int)
    daily['DayName'] = daily['Date'].dt.day_name()
    daily['MonthName'] = daily['Date'].dt.month_name()
    daily['DayOfYear'] = daily['Date'].dt.dayofyear
    return daily

def add_lag_features(daily):
    print('[INFO] Adding lag features...')
    lag_periods = [1, 7, 14, 30]
    for lag in lag_periods:
        daily[f'Sales_Lag{lag}'] = daily['Sales'].shift(lag)
    return daily

def add_rolling_features(daily):
    print('[INFO] Adding rolling statistics...')
    windows = [7, 14, 30]
    for w in windows:
        daily[f'Sales_RollingMean{w}'] = daily['Sales'].rolling(window=w).mean()
        daily[f'Sales_RollingStd{w}'] = daily['Sales'].rolling(window=w).std()
        daily[f'Sales_RollingMin{w}'] = daily['Sales'].rolling(window=w).min()
        daily[f'Sales_RollingMax{w}'] = daily['Sales'].rolling(window=w).max()
    daily['Sales_ExpandingMean'] = daily['Sales'].expanding().mean()
    daily['Sales_PctChange'] = daily['Sales'].pct_change().replace([np.inf, -np.inf], 0)
    return daily

def add_trend_features(daily):
    print('[INFO] Adding trend features...')
    daily['Trend'] = range(len(daily))
    daily['Sin_DayOfYear'] = np.sin(2 * np.pi * daily['DayOfYear'] / 365.25)
    daily['Cos_DayOfYear'] = np.cos(2 * np.pi * daily['DayOfYear'] / 365.25)
    daily['Sin_DayOfWeek'] = np.sin(2 * np.pi * daily['DayOfWeek'] / 7)
    daily['Cos_DayOfWeek'] = np.cos(2 * np.pi * daily['DayOfWeek'] / 7)
    return daily

def save_aggregations(daily, weekly, monthly):
    print(f'\n[INFO] Saving aggregated data...')
    daily.to_csv(DAILY_CSV, index=False)
    weekly.to_csv(WEEKLY_CSV, index=False)
    monthly.to_csv(MONTHLY_CSV, index=False)
    print(f'[SUCCESS] All aggregations saved')
if __name__ == '__main__':
    df = load_cleaned_data()
    daily = create_daily_aggregation(df)
    weekly = create_weekly_aggregation(df)
    monthly = create_monthly_aggregation(df)
    daily = add_time_features(daily)
    daily = add_lag_features(daily)
    daily = add_rolling_features(daily)
    daily = add_trend_features(daily)
    save_aggregations(daily, weekly, monthly)
    print('\n' + '=' * 60)
    print('  STEP 3 COMPLETE ✓')
    print('=' * 60)
