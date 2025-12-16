"""
EDA Script for AQI Predictor
Performs exploratory data analysis to identify trends and patterns.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import settings
from src.feature_store import FeatureStore
from src.data_fetch import fetch_air_quality
from src.feature_engineering import build_features

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def main():
    print("=" * 60)
    print("AQI Predictor - Exploratory Data Analysis")
    print("=" * 60)
    
    # Load data
    store = FeatureStore(settings.feature_store_path)
    df = store.load()
    
    if df is None or df.empty:
        print("Feature store empty, fetching sample data...")
        raw = fetch_air_quality(settings.default_city, past_days=2, forecast_days=1)
        df = build_features(raw)
        store.append(df)
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Basic statistics
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(df.describe())
    
    # Missing values
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
    
    # Temporal patterns
    if 'target_aqi' in df.columns and 'hour' in df.columns:
        print("\n" + "=" * 60)
        print("TEMPORAL PATTERNS")
        print("=" * 60)
        hourly_avg = df.groupby('hour')['target_aqi'].mean()
        print(f"Peak hour (highest AQI): {hourly_avg.idxmax()} ({hourly_avg.max():.1f})")
        print(f"Lowest hour (lowest AQI): {hourly_avg.idxmin()} ({hourly_avg.min():.1f})")
        
        if 'dayofweek' in df.columns:
            df['day_name'] = df['time'].dt.day_name()
            daily_avg = df.groupby('day_name')['target_aqi'].mean()
            print(f"Highest day: {daily_avg.idxmax()} ({daily_avg.max():.1f})")
            print(f"Lowest day: {daily_avg.idxmin()} ({daily_avg.min():.1f})")
    
    # Correlation analysis
    if 'target_aqi' in df.columns:
        print("\n" + "=" * 60)
        print("CORRELATION WITH TARGET AQI")
        print("=" * 60)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        target_corr = corr_matrix['target_aqi'].drop('target_aqi').abs().sort_values(ascending=False)
        print("Top features correlated with AQI:")
        for feature, corr in target_corr.head(10).items():
            print(f"  {feature}: {corr:.3f}")
    
    # AQI distribution
    if 'target_aqi' in df.columns:
        print("\n" + "=" * 60)
        print("AQI DISTRIBUTION STATISTICS")
        print("=" * 60)
        print(f"Mean: {df['target_aqi'].mean():.2f}")
        print(f"Median: {df['target_aqi'].median():.2f}")
        print(f"Std: {df['target_aqi'].std():.2f}")
        print(f"Min: {df['target_aqi'].min():.2f}")
        print(f"Max: {df['target_aqi'].max():.2f}")
        
        high_aqi = df[df['target_aqi'] >= 100]
        print(f"\nHazardous periods (AQI >= 100): {len(high_aqi)} ({len(high_aqi)/len(df)*100:.1f}%)")
    
    # Create visualization
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'target_aqi' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # AQI over time
        df_plot = df.set_index('time').sort_index()
        axes[0, 0].plot(df_plot.index, df_plot['target_aqi'], linewidth=1.5)
        axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=150, color='orange', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('AQI Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('AQI')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hourly pattern
        if 'hour' in df.columns:
            hourly_avg = df.groupby('hour')['target_aqi'].mean()
            axes[0, 1].bar(hourly_avg.index, hourly_avg.values, alpha=0.7)
            axes[0, 1].set_title('Average AQI by Hour')
            axes[0, 1].set_xlabel('Hour')
            axes[0, 1].set_ylabel('Average AQI')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Distribution
        axes[1, 0].hist(df['target_aqi'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['target_aqi'].mean(), color='r', linestyle='--', label='Mean')
        axes[1, 0].set_title('AQI Distribution')
        axes[1, 0].set_xlabel('AQI')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation heatmap (top features)
        if 'target_aqi' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            top_features = list(target_corr.head(8).index) + ['target_aqi']
            top_features = [f for f in top_features if f in df.columns]
            corr_subset = df[top_features].corr()
            sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, ax=axes[1, 1], cbar_kws={"shrink": 0.8})
            axes[1, 1].set_title('Top Feature Correlations')
        
        plt.tight_layout()
        plt.savefig(output_dir / "eda_summary.png", dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_dir / 'eda_summary.png'}")
    
    print("\n" + "=" * 60)
    print("EDA Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

