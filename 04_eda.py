import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
CLEANED_CSV = os.path.join(DATA_DIR, 'cleaned_superstore.csv')
DAILY_CSV = os.path.join(DATA_DIR, 'daily_sales.csv')
MONTHLY_CSV = os.path.join(DATA_DIR, 'monthly_sales.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({'figure.facecolor': '#0f0f23', 'axes.facecolor': '#1a1a2e', 'axes.edgecolor': '#3a3a5c', 'axes.labelcolor': '#e0e0e0', 'text.color': '#e0e0e0', 'xtick.color': '#b0b0c8', 'ytick.color': '#b0b0c8', 'grid.color': '#2a2a4a', 'grid.alpha': 0.5, 'font.family': 'sans-serif', 'font.size': 11, 'figure.dpi': 150})
ACCENT = '#00d4ff'
ACCENT2 = '#ff6b6b'
ACCENT3 = '#51cf66'
BAR_PALETTE = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a', '#fee140', '#30cfd0']

def load_data():
    print('=' * 60)
    print('  STEP 4: EXPLORATORY DATA ANALYSIS')
    print('=' * 60)
    df = pd.read_csv(CLEANED_CSV, parse_dates=['Order Date', 'Ship Date'])
    daily = pd.read_csv(DAILY_CSV, parse_dates=['Date'])
    monthly = pd.read_csv(MONTHLY_CSV, parse_dates=['Month'])
    return (df, daily, monthly)

def plot_daily_sales_trend(daily):
    print('\n[PLOT] Daily sales trend...')
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(daily['Date'], daily['Sales'], alpha=0.15, color=ACCENT)
    ax.plot(daily['Date'], daily['Sales'], alpha=0.4, linewidth=0.5, color=ACCENT, label='Daily Sales')
    rolling_30 = daily['Sales'].rolling(30).mean()
    ax.plot(daily['Date'], rolling_30, color=ACCENT3, linewidth=2, label='30-Day Moving Average')
    ax.set_title('Daily Sales Trend — Superstore', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(loc='upper left', framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_daily_sales_trend.png'), bbox_inches='tight')
    plt.close(fig)

def plot_monthly_sales(monthly):
    print('[PLOT] Monthly sales bar chart...')
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(monthly['Month'], monthly['Sales'], width=20, color=BAR_PALETTE[:len(monthly) % 10], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title('Monthly Sales — Superstore', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.xticks(rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_monthly_sales.png'), bbox_inches='tight')
    plt.close(fig)

def plot_category_sales(df):
    print('[PLOT] Sales by Category...')
    cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cat_sales.index, cat_sales.values, color=[ACCENT, ACCENT2, ACCENT3], alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, cat_sales.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f'${val / 1000:,.0f}k', ha='center', va='bottom', fontsize=10, color='#e0e0e0')
    ax.set_title('Total Sales by Category', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Sales ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '03_category_sales.png'), bbox_inches='tight')
    plt.close(fig)

def plot_top_states(df):
    print('[PLOT] Top 10 states by sales...')
    state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(range(len(state_sales)), state_sales.values, color=BAR_PALETTE[:len(state_sales)], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(state_sales)))
    ax.set_yticklabels(state_sales.index, fontsize=10)
    ax.invert_yaxis()
    for bar, val in zip(bars, state_sales.values):
        ax.text(val + state_sales.max() * 0.01, bar.get_y() + bar.get_height() / 2, f'${val:,.0f}', va='center', fontsize=9, color='#e0e0e0')
    ax.set_title('Top 10 States by Total Sales', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Total Sales ($)')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '04_top_states.png'), bbox_inches='tight')
    plt.close(fig)

def plot_correlation_heatmap(daily):
    print('[PLOT] Correlation heatmap...')
    numeric_cols = ['Sales', 'Profit', 'TotalQuantity', 'NumOrders', 'NumCustomers']
    available = [c for c in numeric_cols if c in daily.columns]
    corr = daily[available].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt='.2f', square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 10, 'color': '#e0e0e0'})
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05_correlation_heatmap.png'), bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    df, daily, monthly = load_data()
    plot_daily_sales_trend(daily)
    plot_monthly_sales(monthly)
    plot_category_sales(df)
    plot_top_states(df)
    plot_correlation_heatmap(daily)
    print(f'\n[SUCCESS] EDA plots saved to {OUTPUT_DIR}/')
    print('\n' + '=' * 60)
    print('  STEP 4 COMPLETE ✓')
    print('=' * 60)
