"""
STAGE 1: ENCODING & MODELLING DATASET PREPARATION
Load cleaned data → Encode categoricals → Verify → Save
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("STAGE 1: ENCODING & MODELLING DATASET PREPARATION")
print("="*70)

# ============================================================================
# 1. SETUP: CREATE OUTPUT FOLDER
# ============================================================================

output_dir = 'Stage1_Encoding_Output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n✓ Created output directory: {output_dir}")

# ============================================================================
# 2. LOAD CLEANED DATASET
# ============================================================================

print("\n" + "="*70)
print("LOADING CLEANED DATASET")
print("="*70)

df = pd.read_csv('cleaned_uber_lyft_dataset.csv')

print(f"\n✓ Dataset loaded")
print(f"  Observations: {len(df):,}")
print(f"  Variables: {len(df.columns)}")
print(f"  Columns: {list(df.columns)}")

# Display first few rows
print(f"\nFirst 3 rows:")
print(df.head(3))

# Check data types
print(f"\nData types (before encoding):")
print(df.dtypes)

# ============================================================================
# 3. VERIFY REQUIRED VARIABLES
# ============================================================================

print("\n" + "="*70)
print("VERIFYING REQUIRED VARIABLES")
print("="*70)

required_vars = ['price', 'surge_multiplier', 'distance', 'hour', 'temperature', 'cab_type', 'source', 'destination']

print(f"\nRequired variables for modelling:")
for var in required_vars:
    if var in df.columns:
        print(f"  ✓ {var}")
    else:
        print(f"  ✗ {var} — MISSING!")
        
missing = [v for v in required_vars if v not in df.columns]
if missing:
    raise ValueError(f"Missing required variables: {missing}")

# ============================================================================
# 4. ENCODE CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*70)
print("ENCODING CATEGORICAL VARIABLES")
print("="*70)

# Create copy for encoding
df_model = df.copy()

# Remove 'id' column (not needed for modelling)
if 'id' in df_model.columns:
    df_model = df_model.drop('id', axis=1)
    print(f"\n✓ Dropped 'id' column (not a predictor)")

# Identify categorical variables to encode
categorical_vars = ['cab_type', 'hour', 'source', 'destination']

print(f"\nCategorical variables to encode: {categorical_vars}")

for var in categorical_vars:
    print(f"\n  {var}:")
    print(f"    Unique values: {df_model[var].nunique()}")
    print(f"    Values: {sorted(df_model[var].unique())}")

# One-hot encode categorical variables, dropping first category (reference)
print(f"\nPerforming one-hot encoding with drop_first=True (to avoid dummy variable trap)...")
df_encoded = pd.get_dummies(df_model, columns=categorical_vars, drop_first=True, dtype=int)

print(f"\n✓ Encoding complete")
print(f"  Original columns: {len(df_model.columns)}")
print(f"  Encoded columns: {len(df_encoded.columns)}")
print(f"  New dummy variables created: {len(df_encoded.columns) - len(df_model.columns)}")

# ============================================================================
# 5. VERIFY DATA TYPES (ALL MUST BE NUMERIC)
# ============================================================================

print("\n" + "="*70)
print("VERIFYING DATA TYPES (ALL MUST BE NUMERIC)")
print("="*70)

print(f"\nData types after encoding:")
print(df_encoded.dtypes)

# Check for non-numeric columns
non_numeric = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"\n✗ WARNING: Non-numeric columns found: {non_numeric}")
    raise ValueError(f"Non-numeric columns after encoding: {non_numeric}")
else:
    print(f"\n✓ All columns are numeric")

# ============================================================================
# 6. DISPLAY ENCODED DATASET STRUCTURE
# ============================================================================

print("\n" + "="*70)
print("ENCODED DATASET STRUCTURE")
print("="*70)

print(f"\nEncoded columns ({len(df_encoded.columns)} total):")
for i, col in enumerate(df_encoded.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nFirst 5 rows of encoded dataset:")
print(df_encoded.head(5))

# ============================================================================
# 7. SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS (CONTINUOUS VARIABLES)")
print("="*70)

continuous_cols = ['price', 'surge_multiplier', 'distance', 'temperature']
print(f"\n{df_encoded[continuous_cols].describe().round(2)}")

# Check for missing values
print(f"\n" + "="*70)
print("MISSING VALUE CHECK")
print("="*70)

missing_count = df_encoded.isnull().sum().sum()
if missing_count == 0:
    print(f"\n✓ No missing values in encoded dataset")
else:
    print(f"\n✗ WARNING: {missing_count} missing values found")
    print(df_encoded.isnull().sum()[df_encoded.isnull().sum() > 0])

# ============================================================================
# 8. SAVE ENCODED DATASET
# ============================================================================

print("\n" + "="*70)
print("SAVING ENCODED DATASET")
print("="*70)

output_path = f'{output_dir}/stage1_encoded_modelling_dataset.csv'
df_encoded.to_csv(output_path, index=False)
print(f"\n✓ Saved: {output_path}")
print(f"  Observations: {len(df_encoded):,}")
print(f"  Variables: {len(df_encoded.columns)}")

# ============================================================================
# 9. SAVE DATA TYPE REPORT
# ============================================================================

dtype_report = pd.DataFrame({
    'Column': df_encoded.columns,
    'Data_Type': df_encoded.dtypes.astype(str),
    'Non_Null_Count': df_encoded.count(),
    'Null_Count': df_encoded.isnull().sum()
})

dtype_path = f'{output_dir}/stage1_data_types_report.csv'
dtype_report.to_csv(dtype_path, index=False)
print(f"✓ Saved: {dtype_path}")

# ============================================================================
# 10. FINAL VERIFICATION SUMMARY
# ============================================================================

print("\n" + "="*70)
print("STAGE 1 COMPLETE: VERIFICATION SUMMARY")
print("="*70)

print(f"\n✓ Dataset Shape: {df_encoded.shape[0]:,} observations × {df_encoded.shape[1]} variables")
print(f"✓ All columns numeric: {df_encoded.select_dtypes(include=[np.number]).shape[1] == len(df_encoded.columns)}")
print(f"✓ Missing values: {df_encoded.isnull().sum().sum()}")
print(f"✓ Categorical encoding: Complete (dummy variable trap avoided with drop_first=True)")
print(f"✓ Reference categories: cab_type=Lyft, hour=0, source=Back Bay, destination=Back Bay")

print(f"\nModelling variables ({len(df_encoded.columns)} total):")
print(f"  Dependent (1): price")
print(f"  Predictors ({len(df_encoded.columns)-1}): {', '.join([c for c in df_encoded.columns if c != 'price'][:5])}... (see saved report)")

print(f"\nOutput files saved to: {output_dir}/")
print(f"  - stage1_encoded_modelling_dataset.csv")
print(f"  - stage1_data_types_report.csv")

print("\n" + "="*70)
print("✓ STAGE 1 READY FOR VERIFICATION")
print("✓ Ready to proceed to STAGE 2 (Train/Test Split)")
print("="*70)