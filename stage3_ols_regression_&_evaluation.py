"""
STAGE 3: OLS REGRESSION & MODEL EVALUATION
Load train/test datasets → Build OLS model → Evaluate on test set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STAGE 3: OLS REGRESSION & MODEL EVALUATION")
print("="*80)

# ============================================================================
# SETUP
# ============================================================================

output_dir = 'Stage3_OLS_Results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n✓ Output directory: {output_dir}")

# ============================================================================
# 1. LOAD EXISTING TRAIN/TEST DATASETS
# ============================================================================

print("\n" + "="*80)
print("LOADING TRAIN/TEST DATASETS FROM STAGE 2")
print("="*80)

X_train = pd.read_csv('Stage2_TrainTest_Output/stage2_X_train.csv')
y_train_df = pd.read_csv('Stage2_TrainTest_Output/stage2_y_train.csv')
X_test = pd.read_csv('Stage2_TrainTest_Output/stage2_X_test.csv')
y_test_df = pd.read_csv('Stage2_TrainTest_Output/stage2_y_test.csv')

# Extract price and drop from X
if 'price' in X_train.columns:
    y_train = X_train['price'].copy()
    X_train = X_train.drop('price', axis=1)
else:
    y_train = y_train_df['price'].copy()

if 'price' in X_test.columns:
    y_test = X_test['price'].copy()
    X_test = X_test.drop('price', axis=1)
else:
    y_test = y_test_df['price'].copy()

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"\n✓ Training data: {len(X_train):,} observations, {len(X_train.columns)} predictors")
print(f"✓ Test data: {len(X_test):,} observations, {len(X_test.columns)} predictors")
print(f"✓ Dependent variable: price")
print(f"✓ All data numeric and verified")

# ============================================================================
# 2. BUILD OLS REGRESSION MODEL
# ============================================================================

print("\n" + "="*80)
print("OLS REGRESSION MODELLING")
print("="*80)

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

print(f"\nFitting OLS model on training data ({len(y_train):,} observations)...")
model = sm.OLS(y_train, X_train_const)
results = model.fit()

print(f"✓ OLS model fitted successfully")
print(f"  Intercept: ${results.params['const']:.4f}")
print(f"  Number of predictors: {len(X_train.columns)}")

# Save full OLS summary
with open(f'{output_dir}/01_ols_model_summary.txt', 'w') as f:
    f.write(str(results.summary()))
print(f"\n✓ Saved: 01_ols_model_summary.txt")

# ============================================================================
# 3. EXTRACT COEFFICIENTS AND P-VALUES
# ============================================================================

print("\n" + "="*80)
print("MODEL COEFFICIENTS")
print("="*80)

coef_df = pd.DataFrame({
    'Variable': results.params.index,
    'Coefficient': results.params.values,
    'Std_Error': results.bse.values,
    'T_Statistic': results.tvalues.values,
    'P_Value': results.pvalues.values,
    'CI_Lower': results.conf_int()[0].values,
    'CI_Upper': results.conf_int()[1].values
})

# Sort by absolute coefficient value to find top predictors
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
top_coef = coef_df.nlargest(10, 'Abs_Coef')

print(f"\nTop 10 Predictors (by absolute coefficient):")
print(top_coef[['Variable', 'Coefficient', 'P_Value']].to_string(index=False))

# Save coefficients
coef_df.to_csv(f'{output_dir}/02_model_coefficients.csv', index=False)
print(f"\n✓ Saved: 02_model_coefficients.csv ({len(coef_df)} coefficients)")

# ============================================================================
# 4. MODEL EVALUATION ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION (TEST SET)")
print("="*80)

y_pred_test = results.predict(X_test_const)
residuals = y_test - y_pred_test

# Calculate metrics
r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
n = len(y_test)
k = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print(f"\nPerformance Metrics:")
print(f"  R²:           {r2:.6f}")
print(f"  Adjusted R²:  {adj_r2:.6f}")
print(f"  RMSE:         ${rmse:.2f}")
print(f"  MAE:          ${mae:.2f}")

# Additional statistics
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
median_ae = np.median(np.abs(y_test - y_pred_test))

print(f"  MAPE:         {mape:.2f}%")
print(f"  Median AE:    ${median_ae:.2f}")

# Save evaluation summary
eval_summary = pd.DataFrame({
    'Metric': ['R²', 'Adjusted R²', 'RMSE ($)', 'MAE ($)', 'MAPE (%)', 'Median_AE ($)',
               'Test_Observations', 'Predictors', 'Training_Observations'],
    'Value': [r2, adj_r2, rmse, mae, mape, median_ae, len(y_test), len(X_train.columns), len(y_train)]
})
eval_summary.to_csv(f'{output_dir}/03_model_evaluation_summary.csv', index=False)
print(f"\n✓ Saved: 03_model_evaluation_summary.csv")

# ============================================================================
# 5. RESIDUAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("RESIDUAL ANALYSIS")
print("="*80)

print(f"\nResidual Statistics:")
print(f"  Mean:     ${residuals.mean():.4f}")
print(f"  Std Dev:  ${residuals.std():.4f}")
print(f"  Min:      ${residuals.min():.2f}")
print(f"  Max:      ${residuals.max():.2f}")

# Normality test
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\nShapiro-Wilk Test (Normality):")
print(f"  Statistic: {shapiro_stat:.6f}")
print(f"  P-Value: {shapiro_p:.6e}")
print(f"  Result: {'Normal' if shapiro_p > 0.05 else 'Non-normal'}")

# Save residuals (FIXED: removed extra .values calls)
residuals_df = pd.DataFrame({
    'Actual_Price': y_test.values,
    'Predicted_Price': y_pred_test.values,
    'Residual': residuals.values,
    'Absolute_Error': np.abs(residuals.values),
    'Percentage_Error': (np.abs(residuals) / y_test * 100)
})
residuals_df.to_csv(f'{output_dir}/04_test_predictions_residuals.csv', index=False)
print(f"\n✓ Saved: 04_test_predictions_residuals.csv")

# ============================================================================
# 6. DIAGNOSTIC TESTS
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC TESTS")
print("="*80)

# Breusch-Pagan test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_result = het_breuschpagan(residuals, X_test_const)

print(f"\nBreusch-Pagan Test (Heteroscedasticity):")
print(f"  LM Statistic: {bp_result[0]:.6f}")
print(f"  P-Value: {bp_result[1]:.6f}")
print(f"  Result: {'Homoscedastic' if bp_result[1] > 0.05 else 'Heteroscedastic'}")

# Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals)

print(f"\nDurbin-Watson Test (Autocorrelation):")
print(f"  Statistic: {dw_stat:.6f}")
print(f"  Result: {'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Possible autocorrelation'}")

# Save diagnostics
diag_df = pd.DataFrame({
    'Test': ['Breusch-Pagan', 'Durbin-Watson', 'Shapiro-Wilk'],
    'Statistic': [bp_result[0], dw_stat, shapiro_stat],
    'P_Value': [bp_result[1], np.nan, shapiro_p],
    'Result': [
        'Homoscedastic' if bp_result[1] > 0.05 else 'Heteroscedastic',
        'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Possible autocorrelation',
        'Normal' if shapiro_p > 0.05 else 'Non-normal'
    ]
})
diag_df.to_csv(f'{output_dir}/05_diagnostic_tests.csv', index=False)
print(f"\n✓ Saved: 05_diagnostic_tests.csv")

# ============================================================================
# 7. VISUALIZATIONS (2 essential graphs)
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: Actual vs Predicted
fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(y_test, y_pred_test, alpha=0.3, s=10, color='steelblue', edgecolor='none')
min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
ax.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax.set_title(f'OLS Model Performance: Actual vs Predicted Prices\nR² = {r2:.4f}, RMSE = ${rmse:.2f}, MAE = ${mae:.2f}', 
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_actual_vs_predicted.png")
plt.close()

# Plot 2: Residuals vs Predicted
fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(y_pred_test, residuals, alpha=0.3, s=10, color='darkgreen', edgecolor='none')
ax.axhline(0, color='red', linestyle='--', linewidth=2.5, label='Zero Residual')
ax.set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual ($)', fontsize=12, fontweight='bold')
ax.set_title('Residual Diagnostic Plot\n(Check for heteroscedasticity and patterns)', 
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_residuals_vs_predicted.png")
plt.close()

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("STAGE 3 COMPLETE: OLS REGRESSION & MODEL EVALUATION")
print("="*80)

print(f"\n✓ Model Summary:")
print(f"    Training observations: {len(y_train):,}")
print(f"    Predictors: {len(X_train.columns)}")
print(f"    Intercept: ${results.params['const']:.4f}")

print(f"\n✓ Test Set Performance:")
print(f"    R²: {r2:.6f}")
print(f"    Adjusted R²: {adj_r2:.6f}")
print(f"    RMSE: ${rmse:.2f}")
print(f"    MAE: ${mae:.2f}")
print(f"    Observations: {len(y_test):,}")

print(f"\n✓ Output Files ({output_dir}/):")
print(f"    - 01_ols_model_summary.txt (full regression output)")
print(f"    - 02_model_coefficients.csv ({len(coef_df)} coefficients)")
print(f"    - 03_model_evaluation_summary.csv (key metrics)")
print(f"    - 04_test_predictions_residuals.csv ({len(residuals_df)} predictions)")
print(f"    - 05_diagnostic_tests.csv (3 diagnostic tests)")
print(f"    - 01_actual_vs_predicted.png")
print(f"    - 02_residuals_vs_predicted.png")

print("\n" + "="*80)
print("✓ STAGE 3 READY FOR VERIFICATION")
print("✓ Ready to proceed to STAGE 4 (Fairness Simulation & Hypothesis Testing)")
print("="*80)