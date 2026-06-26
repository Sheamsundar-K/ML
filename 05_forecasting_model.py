import os
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
DAILY_CSV = os.path.join(DATA_DIR, 'daily_sales.csv')
FORECAST_DAYS = 90
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({'figure.facecolor': '#0f0f23', 'axes.facecolor': '#1a1a2e', 'axes.edgecolor': '#3a3a5c', 'axes.labelcolor': '#e0e0e0', 'text.color': '#e0e0e0', 'xtick.color': '#b0b0c8', 'ytick.color': '#b0b0c8', 'grid.color': '#2a2a4a', 'grid.alpha': 0.5, 'font.size': 11, 'figure.dpi': 150})
ACCENT, ACCENT2 = ('#00d4ff', '#ff6b6b')

def load_data():
    print('=' * 60)
    print('  STEP 5: FORECASTING MODEL (FUTURE PREDICTION)')
    print('=' * 60)
    return pd.read_csv(DAILY_CSV, parse_dates=['Date'])

def train_and_forecast(daily):
    print('\n[INFO] Training Random Forest on entire historical dataset...')
    feature_cols = ['Year', 'Month', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Quarter', 'IsWeekend', 'DayOfYear', 'Trend', 'Sin_DayOfYear', 'Cos_DayOfYear', 'Sin_DayOfWeek', 'Cos_DayOfWeek']
    X_train = daily[feature_cols]
    y_train = daily['Sales']
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    with open(os.path.join(OUTPUT_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    print(f'[INFO] Generating forecast for the next {FORECAST_DAYS} days...')
    last_date = daily['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['DayOfMonth'] = future_df['Date'].dt.day
    future_df['WeekOfYear'] = future_df['Date'].dt.isocalendar().week.astype(int)
    future_df['Quarter'] = future_df['Date'].dt.quarter
    future_df['IsWeekend'] = (future_df['DayOfWeek'] >= 5).astype(int)
    future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
    last_trend = daily['Trend'].max()
    future_df['Trend'] = range(last_trend + 1, last_trend + 1 + FORECAST_DAYS)
    future_df['Sin_DayOfYear'] = np.sin(2 * np.pi * future_df['DayOfYear'] / 365.25)
    future_df['Cos_DayOfYear'] = np.cos(2 * np.pi * future_df['DayOfYear'] / 365.25)
    future_df['Sin_DayOfWeek'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
    future_df['Cos_DayOfWeek'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)
    X_future = future_df[feature_cols]
    future_preds = np.clip(rf_model.predict(X_future), 0, None)
    future_df['Forecast_Sales'] = future_preds
    future_df.to_csv(os.path.join(DATA_DIR, 'future_forecast.csv'), index=False)
    return future_df

def plot_future_forecast(daily, future_df):
    print('\n[PLOT] Visualizing future forecast...')
    fig, ax = plt.subplots(figsize=(15, 6))
    hist_plot = daily.tail(365).copy()
    ax.plot(hist_plot['Date'], hist_plot['Sales'].rolling(7).mean(), color='#ffffff', linewidth=1.5, alpha=0.9, label='Historical Sales (7-day avg)')
    connector_date = hist_plot['Date'].iloc[-1]
    connector_sales = hist_plot['Sales'].rolling(7).mean().iloc[-1]
    plot_dates = [connector_date] + list(future_df['Date'])
    plot_sales = [connector_sales] + list(future_df['Forecast_Sales'].rolling(7, min_periods=1).mean())
    ax.plot(plot_dates, plot_sales, color=ACCENT, linewidth=2.5, linestyle='--', label='Forecasted Sales (7-day avg)')
    ax.axvspan(connector_date, future_df['Date'].max(), color=ACCENT, alpha=0.1)
    ax.axvline(x=connector_date, color='#ffd43b', linestyle='--', alpha=0.7, linewidth=1, label='Today')
    ax.set_title('Future Sales Prediction (Next 90 Days)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(loc='upper left', framealpha=0.8, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '09_future_forecast.png'), bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    daily = load_data()
    future_df = train_and_forecast(daily)
    plot_future_forecast(daily, future_df)
    print('\n' + '=' * 60)
    print('  STEP 5 COMPLETE ✓')
    print('=' * 60)
