"""
Exploratory Data Analysis (EDA) for Uber/Lyft Boston Dataset
Methodology: Characterise data, assess regression assumptions, investigate relationships
Output: Graphs (PNG), statistics (CSV), route analysis (CSV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SETUP: CREATE OUTPUT FOLDER AND LOAD DATA
# ============================================================================

output_dir = 'EDA_Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Created output directory: {output_dir}")

# Load cleaned dataset
df = pd.read_csv('cleaned_uber_lyft_dataset.csv')
print(f"✓ Dataset loaded: {len(df):,} observations, {len(df.columns)} variables")
print(f"  Variables: {list(df.columns)}\n")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================

print("="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

# Continuous variables
continuous_vars = ['price', 'surge_multiplier', 'distance', 'temperature']

desc_stats = df[continuous_vars].describe().T
desc_stats['variance'] = df[continuous_vars].var()
desc_stats['skewness'] = df[continuous_vars].skew()
desc_stats['kurtosis'] = df[continuous_vars].kurtosis()

print("\nContinuous Variables Summary:")
print(desc_stats.round(4))
desc_stats.to_csv(f'{output_dir}/descriptive_statistics.csv')
print(f"✓ Saved: descriptive_statistics.csv")

# Categorical variables
print("\n" + "="*70)
print("CATEGORICAL VARIABLES")
print("="*70)
print(f"\nCab Type Distribution:")
print(df['cab_type'].value_counts())
print(f"\nHour Distribution (sample):")
print(df['hour'].value_counts().sort_index().head(10))

# ============================================================================
# 3. DISTRIBUTIONAL ANALYSIS (Histograms + Density Plots)
# ============================================================================

print("\n" + "="*70)
print("DISTRIBUTIONAL ANALYSIS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Continuous Variables', fontsize=16, fontweight='bold')

for idx, var in enumerate(continuous_vars):
    ax = axes[idx // 2, idx % 2]
    
    # Histogram with density overlay
    ax.hist(df[var], bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Histogram')
    
    # Density plot
    df[var].plot(kind='density', ax=ax, color='darkred', linewidth=2, label='Density')
    
    # Normality test (Shapiro-Wilk)
    if len(df) <= 5000:
        stat, p_value = stats.shapiro(df[var].dropna())
    else:
        stat, p_value = stats.kstest(df[var], 'norm', args=(df[var].mean(), df[var].std()))
    
    ax.set_title(f'{var.capitalize()} (Normality p-value: {p_value:.4f})', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_distribution_analysis.png")
plt.close()

# ============================================================================
# 4. SCATTER PLOTS: Continuous Predictors vs Price
# ============================================================================

print("\nCreating scatter plots...")

scatter_vars = ['distance', 'surge_multiplier', 'temperature']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Bivariate Relationships: Predictors vs Price', fontsize=14, fontweight='bold')

for idx, var in enumerate(scatter_vars):
    ax = axes[idx]
    ax.scatter(df[var], df['price'], alpha=0.3, s=10, color='steelblue')
    
    # Add trend line
    z = np.polyfit(df[var].dropna(), df['price'][df[var].notna()], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[var].min(), df[var].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend')
    
    # Calculate correlation
    corr = df[var].corr(df['price'])
    ax.set_title(f'{var.capitalize()} vs Price\n(r = {corr:.3f})', fontweight='bold')
    ax.set_xlabel(var.capitalize())
    ax.set_ylabel('Price ($)')
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/02_scatter_continuous_predictors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_scatter_continuous_predictors.png")
plt.close()

# ============================================================================
# 5. SCATTER PLOT: Hour vs Price
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(df['hour'], df['price'], alpha=0.3, s=15, color='darkgreen')

# Add mean price by hour
hourly_mean = df.groupby('hour')['price'].mean()
ax.plot(hourly_mean.index, hourly_mean.values, 'r-', linewidth=3, marker='o', label='Mean Price by Hour')

ax.set_title('Hour of Day vs Price', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day (0-23)')
ax.set_ylabel('Price ($)')
ax.set_xticks(range(0, 24))
ax.grid(alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_scatter_hour_vs_price.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_scatter_hour_vs_price.png")
plt.close()

# ============================================================================
# 6. BOXPLOTS: Price by Categorical Variables
# ============================================================================

print("Creating boxplots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Price Distribution by Categorical Variables', fontsize=14, fontweight='bold')

# Price by Cab Type
ax = axes[0]
df.boxplot(column='price', by='cab_type', ax=ax)
ax.set_title('Price by Cab Type', fontweight='bold')
ax.set_xlabel('Cab Type')
ax.set_ylabel('Price ($)')
ax.grid(alpha=0.3)

# Price by Hour (grouped into bins for clarity)
ax = axes[1]
df.boxplot(column='price', by='hour', ax=ax)
ax.set_title('Price by Hour of Day', fontweight='bold')
ax.set_xlabel('Hour (0-23)')
ax.set_ylabel('Price ($)')
ax.grid(alpha=0.3)

plt.suptitle('')  # Remove automatic title
plt.tight_layout()
plt.savefig(f'{output_dir}/04_boxplots_categorical.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_boxplots_categorical.png")
plt.close()

# ============================================================================
# 7. CORRELATION ANALYSIS & HEATMAP
# ============================================================================

print("Creating correlation heatmap...")

# Correlation matrix for continuous variables
corr_matrix = df[continuous_vars].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Correlation Matrix: Continuous Variables\n(Multicollinearity Screening)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_correlation_heatmap.png")
plt.close()

print("\nCorrelation Matrix:")
print(corr_matrix.round(4))

# ============================================================================
# 8. ROUTE-LEVEL ANALYSIS (Source + Destination)
# ============================================================================

print("\n" + "="*70)
print("ROUTE-LEVEL ANALYSIS")
print("="*70)

# Create route identifier
df['route'] = df['source'] + ' → ' + df['destination']

# Calculate price statistics by route
route_stats = df.groupby('route')['price'].agg([
    ('observation_count', 'count'),
    ('mean_price', 'mean'),
    ('median_price', 'median'),
    ('std_price', 'std'),
    ('min_price', 'min'),
    ('max_price', 'max'),
    ('price_range', lambda x: x.max() - x.min())
]).round(2)

route_stats = route_stats.sort_values('observation_count', ascending=False)

print(f"\nTotal unique routes: {len(route_stats)}")
print(f"\nTop 10 Routes by Observation Count:")
print(route_stats.head(10))

route_stats.to_csv(f'{output_dir}/route_level_analysis.csv')
print(f"\n✓ Saved: route_level_analysis.csv ({len(route_stats)} routes)")

# Visualize route-level price variance
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Route-Level Price Variation (Fairness Proxy)', fontsize=14, fontweight='bold')

# Top 15 routes by frequency - std deviation
ax = axes[0]
top_routes = route_stats.head(15)
ax.barh(range(len(top_routes)), top_routes['std_price'], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top_routes)))
ax.set_yticklabels(top_routes.index, fontsize=9)
ax.set_xlabel('Price Standard Deviation ($)')
ax.set_title('Price Variance (Std Dev) in Top 15 Busiest Routes', fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Price range by route (Top 15)
ax = axes[1]
ax.barh(range(len(top_routes)), top_routes['price_range'], color='darkgreen', alpha=0.7)
ax.set_yticks(range(len(top_routes)))
ax.set_yticklabels(top_routes.index, fontsize=9)
ax.set_xlabel('Price Range ($) [Max - Min]')
ax.set_title('Price Range (Max - Min) in Top 15 Busiest Routes', fontweight='bold')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}/06_route_level_price_variation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_route_level_price_variation.png")
plt.close()

# ============================================================================
# 9. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("EDA SUMMARY")
print("="*70)
print(f"\n✓ Dataset: {len(df):,} observations")
print(f"✓ Continuous variables: {', '.join(continuous_vars)}")
print(f"✓ Categorical variables: cab_type, hour, source, destination")
print(f"✓ Unique routes identified: {len(route_stats)}")
print(f"\n✓ All outputs saved to: {output_dir}/")
print("\nFiles created:")
print("  - descriptive_statistics.csv")
print("  - route_level_analysis.csv")
print("  - 01_distribution_analysis.png")
print("  - 02_scatter_continuous_predictors.png")
print("  - 03_scatter_hour_vs_price.png")
print("  - 04_boxplots_categorical.png")
print("  - 05_correlation_heatmap.png")
print("  - 06_route_level_price_variation.png")

print("\n" + "="*70)
print("✓ EDA COMPLETE")
print("="*70)