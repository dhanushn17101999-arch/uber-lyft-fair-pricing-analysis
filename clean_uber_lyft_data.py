import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os

INPUT_FILE_PATH = "uber_lyft_boston.csv"
OUTPUT_FILE_PATH = "cleaned_uber_lyft_dataset.csv"

COLUMNS_REQUIRED_FOR_ANALYSIS = [
    "id",
    "cab_type",
    "source",
    "destination",
    "price",
    "distance",
    "surge_multiplier",
    "hour",
    "temperature",
]

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Please make sure the "
            f"dataset CSV is in the same folder as this script, or "
            f"update INPUT_FILE_PATH at the top of the script."
        )

    print(f"Loading dataset from: {file_path}")
    dataframe = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {dataframe.shape[0]} rows, "
          f"{dataframe.shape[1]} columns.\n")
    return dataframe

def inspect_dataset(dataframe):
    print("=" * 60)
    print("DATASET INSPECTION (BEFORE CLEANING)")
    print("=" * 60)

    print(f"\nNumber of rows: {dataframe.shape[0]}")
    print(f"Number of columns: {dataframe.shape[1]}")

    print("\nColumn data types:")
    print(dataframe.dtypes)

    print("\nMissing values per column (top 10 shown):")
    missing_counts = dataframe.isnull().sum().sort_values(ascending=False)
    print(missing_counts.head(10))

    print("\nNumber of fully duplicated rows:", dataframe.duplicated().sum())

    print("\nFirst 5 rows of the dataset:")
    print(dataframe.head())
    print("=" * 60 + "\n")

def select_required_columns(dataframe, required_columns):
    missing_from_data = [c for c in required_columns if c not in dataframe.columns]

    if missing_from_data:
        raise KeyError(
            f"The following required columns were not found in the "
            f"dataset: {missing_from_data}. Please check your CSV file "
            f"has the expected column names."
        )

    trimmed_dataframe = dataframe[required_columns].copy()

    print(f"Kept {len(required_columns)} required columns: {required_columns}")
    print(f"Dropped {dataframe.shape[1] - len(required_columns)} unnecessary "
          f"columns (e.g. detailed weather fields not used in the model).\n")

    return trimmed_dataframe

def handle_missing_values(dataframe):
    rows_before = len(dataframe)

    missing_price_count = dataframe["price"].isnull().sum()

    print(f"Rows with missing price: {missing_price_count} "
          f"({missing_price_count / rows_before:.2%} of the dataset).")
    print("Following the methodology's recommendation (Option A), these "
          "rows will be removed rather than imputed.")

    cleaned_dataframe = dataframe.dropna(
        subset=[
            "price",
            "distance",
            "surge_multiplier",
            "hour",
            "temperature",
            "cab_type",
            "source",
            "destination",
        ]
    ).copy()

    rows_after = len(cleaned_dataframe)

    print(f"Rows removed due to missing values: {rows_before - rows_after}")
    print(f"Rows remaining: {rows_after}\n")

    return cleaned_dataframe

def remove_duplicate_records(dataframe):
    rows_before = len(dataframe)

    columns_to_check = [c for c in dataframe.columns if c != "id"]

    deduplicated_dataframe = dataframe.drop_duplicates(
        subset=columns_to_check
    ).copy()

    rows_after = len(deduplicated_dataframe)

    print(f"Duplicate rows found and removed: {rows_before - rows_after}")
    print(f"Rows remaining: {rows_after}\n")

    return deduplicated_dataframe

def convert_data_types(dataframe):
    dataframe = dataframe.copy()

    dataframe["price"] = pd.to_numeric(dataframe["price"], errors="coerce")
    dataframe["distance"] = pd.to_numeric(dataframe["distance"], errors="coerce")
    dataframe["surge_multiplier"] = pd.to_numeric(
        dataframe["surge_multiplier"], errors="coerce"
    )
    dataframe["temperature"] = pd.to_numeric(
        dataframe["temperature"], errors="coerce"
    )

    dataframe["hour"] = pd.to_numeric(
        dataframe["hour"], errors="coerce"
    ).astype("Int64")

    dataframe["cab_type"] = dataframe["cab_type"].astype("category")
    dataframe["source"] = dataframe["source"].astype("category")
    dataframe["destination"] = dataframe["destination"].astype("category")
    dataframe["id"] = dataframe["id"].astype(str)

    rows_before = len(dataframe)

    dataframe = dataframe.dropna(
        subset=["price", "distance", "surge_multiplier", "hour", "temperature"]
    ).copy()

    rows_after = len(dataframe)

    if rows_before != rows_after:
        print(f"Rows removed because a value could not be converted to the "
              f"correct type: {rows_before - rows_after}")

    print("Data types after conversion:")
    print(dataframe.dtypes)
    print()

    return dataframe

def check_invalid_values(dataframe):
    rows_before = len(dataframe)
    dataframe = dataframe.copy()

    invalid_price_mask = dataframe["price"] <= 0
    invalid_distance_mask = dataframe["distance"] <= 0

    invalid_surge_mask = (
        (dataframe["surge_multiplier"] < 1.0) |
        (dataframe["surge_multiplier"] > 3.0)
    )

    invalid_hour_mask = (
        (dataframe["hour"] < 0) |
        (dataframe["hour"] > 23)
    )

    invalid_cab_type_mask = ~dataframe["cab_type"].isin(["Uber", "Lyft"])

    invalid_rows_mask = (
        invalid_price_mask
        | invalid_distance_mask
        | invalid_surge_mask
        | invalid_hour_mask
        | invalid_cab_type_mask
    )

    print(f"Rows with an impossible price (<= 0): {invalid_price_mask.sum()}")
    print(f"Rows with an impossible distance (<= 0): {invalid_distance_mask.sum()}")
    print(f"Rows with surge multiplier outside 1.0-3.0: {invalid_surge_mask.sum()}")
    print(f"Rows with hour outside 0-23: {invalid_hour_mask.sum()}")
    print(f"Rows with an unrecognised cab type: {invalid_cab_type_mask.sum()}")

    cleaned_dataframe = dataframe.loc[~invalid_rows_mask].copy()

    rows_after = len(cleaned_dataframe)

    print(f"Total invalid rows removed: {rows_before - rows_after}")
    print(f"Rows remaining: {rows_after}\n")

    cleaned_dataframe = cleaned_dataframe.reset_index(drop=True)

    return cleaned_dataframe

def save_cleaned_dataset(dataframe, output_path):
    dataframe.to_csv(output_path, index=False)

    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Final shape: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns.\n")

def main():
    print("\n" + "#" * 60)
    print("# UBER/LYFT BOSTON DATASET -- DATA CLEANING PIPELINE")
    print("#" * 60 + "\n")

    raw_dataframe = load_dataset(INPUT_FILE_PATH)

    inspect_dataset(raw_dataframe)

    trimmed_dataframe = select_required_columns(
        raw_dataframe, COLUMNS_REQUIRED_FOR_ANALYSIS
    )

    dataframe_no_missing = handle_missing_values(trimmed_dataframe)

    dataframe_no_duplicates = remove_duplicate_records(dataframe_no_missing)

    dataframe_correct_types = convert_data_types(dataframe_no_duplicates)

    final_dataframe = check_invalid_values(dataframe_correct_types)

    save_cleaned_dataset(final_dataframe, OUTPUT_FILE_PATH)

    print("#" * 60)
    print("# CLEANING COMPLETE")
    print("#" * 60 + "\n")

if __name__ == "__main__":
    main()
