"""
STAGE 2: 80:20 TRAIN/TEST SPLIT
Load encoded dataset → Separate X and y → Split (80:20) → Save & Verify
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("STAGE 2: 80:20 TRAIN/TEST SPLIT")
print("="*70)

# ============================================================================
# 1. SETUP: CREATE OUTPUT FOLDER
# ============================================================================

output_dir = 'Stage2_TrainTest_Output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n✓ Created output directory: {output_dir}")

# ============================================================================
# 2. LOAD ENCODED DATASET FROM STAGE 1
# ============================================================================

print("\n" + "="*70)
print("LOADING ENCODED DATASET FROM STAGE 1")
print("="*70)

encoded_path = 'Stage1_Encoding_Output/stage1_encoded_modelling_dataset.csv'
df_encoded = pd.read_csv(encoded_path)

print(f"\n✓ Encoded dataset loaded from Stage 1")
print(f"  Observations: {len(df_encoded):,}")
print(f"  Variables: {len(df_encoded.columns)}")

# Verify all columns are numeric
non_numeric = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    raise ValueError(f"Non-numeric columns found: {non_numeric}")
else:
    print(f"✓ All columns verified as numeric")

# ============================================================================
# 3. SEPARATE DEPENDENT VARIABLE (y) AND PREDICTORS (X)
# ============================================================================

print("\n" + "="*70)
print("SEPARATING DEPENDENT VARIABLE AND PREDICTORS")
print("="*70)

# Dependent variable: price
y = df_encoded['price'].copy()

# Predictors: all columns except price
X = df_encoded.drop('price', axis=1).copy()

print(f"\nDependent Variable (y):")
print(f"  Variable: price")
print(f"  Observations: {len(y):,}")
print(f"  Data type: {y.dtype}")
print(f"  Mean: ${y.mean():.2f}")
print(f"  Std Dev: ${y.std():.2f}")
print(f"  Min: ${y.min():.2f}, Max: ${y.max():.2f}")

print(f"\nIndependent Variables (X):")
print(f"  Number of predictors: {len(X.columns)}")
print(f"  Observations: {len(X):,}")
print(f"  Columns:")
for i, col in enumerate(X.columns, 1):
    print(f"    {i:2d}. {col}")

# ============================================================================
# 4. PERFORM 80:20 TRAIN/TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("PERFORMING 80:20 TRAIN/TEST SPLIT")
print("="*70)

from sklearn.model_selection import train_test_split

random_seed = 42
test_size = 0.20

print(f"\nSplit parameters:")
print(f"  Random seed: {random_seed} (for reproducibility)")
print(f"  Test size: {test_size} (20%)")
print(f"  Training size: {1 - test_size} (80%)")

# Perform split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

print(f"\n✓ Split completed")

# ============================================================================
# 5. VERIFY SPLIT COUNTS
# ============================================================================

print("\n" + "="*70)
print("SPLIT VERIFICATION")
print("="*70)

total_obs = len(X)
train_obs = len(X_train)
test_obs = len(X_test)
train_pct = (train_obs / total_obs) * 100
test_pct = (test_obs / total_obs) * 100

print(f"\nOriginal dataset: {total_obs:,} observations")
print(f"\nTraining set:")
print(f"  Observations: {train_obs:,}")
print(f"  Percentage: {train_pct:.2f}%")
print(f"  Variables: {len(X_train.columns)} predictors + 1 target (price)")

print(f"\nTest set:")
print(f"  Observations: {test_obs:,}")
print(f"  Percentage: {test_pct:.2f}%")
print(f"  Variables: {len(X_test.columns)} predictors + 1 target (price)")

print(f"\nTotal: {train_obs + test_obs:,} observations")
print(f"  (Training {train_pct:.2f}% + Test {test_pct:.2f}% = 100%)")

# Verify no data loss
if train_obs + test_obs != total_obs:
    raise ValueError(f"Data loss detected: {train_obs} + {test_obs} ≠ {total_obs}")
else:
    print(f"\n✓ No data loss: All observations accounted for")

# ============================================================================
# 6. SAVE TRAINING SET
# ============================================================================

print("\n" + "="*70)
print("SAVING TRAINING SET")
print("="*70)

# Combine X_train and y_train for saving
train_data = X_train.copy()
train_data['price'] = y_train.values

train_path = f'{output_dir}/stage2_X_train.csv'
train_data.to_csv(train_path, index=False)
print(f"\n✓ Saved: {train_path}")
print(f"  Observations: {len(train_data):,}")
print(f"  Variables: {len(train_data.columns)}")

# Also save y_train separately for convenience
y_train_path = f'{output_dir}/stage2_y_train.csv'
pd.DataFrame({'price': y_train.values}).to_csv(y_train_path, index=False)
print(f"✓ Saved: {y_train_path}")

# ============================================================================
# 7. SAVE TEST SET
# ============================================================================

print("\n" + "="*70)
print("SAVING TEST SET")
print("="*70)

# Combine X_test and y_test for saving
test_data = X_test.copy()
test_data['price'] = y_test.values

test_path = f'{output_dir}/stage2_X_test.csv'
test_data.to_csv(test_path, index=False)
print(f"\n✓ Saved: {test_path}")
print(f"  Observations: {len(test_data):,}")
print(f"  Variables: {len(test_data.columns)}")

# Also save y_test separately for convenience
y_test_path = f'{output_dir}/stage2_y_test.csv'
pd.DataFrame({'price': y_test.values}).to_csv(y_test_path, index=False)
print(f"✓ Saved: {y_test_path}")

# ============================================================================
# 8. SAVE SPLIT INFORMATION
# ============================================================================

print("\n" + "="*70)
print("SAVING SPLIT INFORMATION")
print("="*70)

split_info = pd.DataFrame({
    'Dataset': ['Training', 'Test', 'Total'],
    'Observations': [train_obs, test_obs, total_obs],
    'Percentage': [f"{train_pct:.2f}%", f"{test_pct:.2f}%", "100.00%"]
})

split_info_path = f'{output_dir}/stage2_split_info.csv'
split_info.to_csv(split_info_path, index=False)
print(f"\n✓ Saved: {split_info_path}")
print(f"\n{split_info}")

# ============================================================================
# 9. VERIFY DATA INTEGRITY
# ============================================================================

print("\n" + "="*70)
print("DATA INTEGRITY VERIFICATION")
print("="*70)

# Check for missing values
train_missing = train_data.isnull().sum().sum()
test_missing = test_data.isnull().sum().sum()

print(f"\nMissing values:")
print(f"  Training set: {train_missing}")
print(f"  Test set: {test_missing}")

if train_missing == 0 and test_missing == 0:
    print(f"  ✓ No missing values")
else:
    print(f"  ✗ WARNING: Missing values detected")

# Check data types
train_non_numeric = train_data.select_dtypes(exclude=[np.number]).columns.tolist()
test_non_numeric = test_data.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\nData types:")
print(f"  Training set all numeric: {len(train_non_numeric) == 0}")
print(f"  Test set all numeric: {len(test_non_numeric) == 0}")

if train_non_numeric:
    print(f"    ✗ Training non-numeric: {train_non_numeric}")
if test_non_numeric:
    print(f"    ✗ Test non-numeric: {test_non_numeric}")

# ============================================================================
# 10. SUMMARY STATISTICS COMPARISON
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS COMPARISON")
print("="*70)

print(f"\nPrice distribution comparison:")
print(f"\n  Original dataset:")
print(f"    Mean: ${y.mean():.2f}")
print(f"    Std:  ${y.std():.2f}")
print(f"    Min:  ${y.min():.2f}")
print(f"    Max:  ${y.max():.2f}")

print(f"\n  Training set:")
print(f"    Mean: ${y_train.mean():.2f}")
print(f"    Std:  ${y_train.std():.2f}")
print(f"    Min:  ${y_train.min():.2f}")
print(f"    Max:  ${y_train.max():.2f}")

print(f"\n  Test set:")
print(f"    Mean: ${y_test.mean():.2f}")
print(f"    Std:  ${y_test.std():.2f}")
print(f"    Min:  ${y_test.min():.2f}")
print(f"    Max:  ${y_test.max():.2f}")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("STAGE 2 COMPLETE: FINAL SUMMARY")
print("="*70)

print(f"\n✓ Train/Test split completed successfully")
print(f"✓ Random seed: {random_seed} (for reproducibility)")
print(f"✓ Training observations: {train_obs:,} (80.0%)")
print(f"✓ Test observations: {test_obs:,} (20.0%)")
print(f"✓ Total observations: {total_obs:,}")
print(f"✓ Predictors: {len(X.columns)}")
print(f"✓ Dependent variable: price")
print(f"✓ All data numeric: True")
print(f"✓ No missing values: True")

print(f"\nOutput files saved to: {output_dir}/")
print(f"  - stage2_X_train.csv (training predictors + target)")
print(f"  - stage2_y_train.csv (training target)")
print(f"  - stage2_X_test.csv (test predictors + target)")
print(f"  - stage2_y_test.csv (test target)")
print(f"  - stage2_split_info.csv (split summary)")

print("\n" + "="*70)
print("✓ STAGE 2 READY FOR VERIFICATION")
print("✓ Ready to proceed to STAGE 3 (OLS Regression & Evaluation)")
print("="*70)