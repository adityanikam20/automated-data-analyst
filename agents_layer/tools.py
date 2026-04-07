import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


# ---------- LOAD DATA ----------
def load_dataset(file_path: str) -> pd.DataFrame:
    lower_path = file_path.lower()

    if lower_path.endswith(".csv"):
        return pd.read_csv(file_path)

    if lower_path.endswith(".xlsx"):
        raw_df = pd.read_excel(file_path, header=None)

        if raw_df.shape[1] > 0:
            first_col = raw_df.iloc[:, 0].astype(str)

            # Special handling for repeated-block sales report
            if first_col.str.contains("Sales for:", na=False).any():
                rows = []
                current_date = None

                for _, row in raw_df.iterrows():
                    first_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""

                    # date row
                    if first_cell.startswith("Sales for:"):
                        current_date = first_cell.replace("Sales for:", "").strip()
                        continue

                    # skip repeated header row
                    if first_cell == "Sales Person":
                        continue

                    # skip total rows / blanks
                    if first_cell == "" or first_cell.lower() == "total" or pd.isna(row.iloc[0]):
                        continue

                    rows.append({
                        "date": current_date,
                        "sales_person": first_cell,
                        "north": row.iloc[1] if len(row) > 1 else None,
                        "east": row.iloc[2] if len(row) > 2 else None,
                        "south": row.iloc[3] if len(row) > 3 else None,
                        "west": row.iloc[4] if len(row) > 4 else None,
                    })

                return pd.DataFrame(rows)

        # Normal Excel file
        return pd.read_excel(file_path)

    raise ValueError("Only CSV and XLSX files are supported.")


# ---------- BASIC PROFILE ----------
def get_basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ---------- CLEAN DATA ----------
def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    issues_found: List[Dict[str, str]] = []
    original_shape = list(df.shape)

    # standardize column names
    old_columns = list(df.columns)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )
    new_columns = list(df.columns)

    for old, new in zip(old_columns, new_columns):
        if old != new:
            issues_found.append({
                "column": old,
                "issue": "Non-standard column name",
                "action_taken": f"Renamed to '{new}'"
            })

    # remove duplicates
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        df = df.drop_duplicates()
        issues_found.append({
            "column": "ALL",
            "issue": "Duplicate rows found",
            "action_taken": f"Removed {duplicate_count} duplicate rows"
        })

    # convert object columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="ignore")
            if not converted.equals(df[col]):
                df[col] = converted
                issues_found.append({
                    "column": col,
                    "issue": "Numeric values stored as text",
                    "action_taken": "Converted to numeric"
                })

    # fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        missing_count = int(df[col].isnull().sum())
        if missing_count > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            issues_found.append({
                "column": col,
                "issue": f"{missing_count} missing numeric values",
                "action_taken": "Filled with median"
            })

    # fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        missing_count = int(df[col].isnull().sum())
        if missing_count > 0:
            mode_series = df[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            issues_found.append({
                "column": col,
                "issue": f"{missing_count} missing categorical values",
                "action_taken": f"Filled with '{fill_value}'"
            })

    final_shape = list(df.shape)

    report = {
        "original_shape": original_shape,
        "final_shape": final_shape,
        "duplicates_removed": duplicate_count,
        "cleaned_columns": list(df.columns),
        "issues_found": issues_found,
        "summary": (
            f"Dataset cleaned successfully. "
            f"Rows: {original_shape[0]} -> {final_shape[0]}, "
            f"Columns: {original_shape[1]} -> {final_shape[1]}."
        )
    }

    return df, report


# ---------- SAVE CLEANED DATASET ----------
def save_cleaned_dataset(df: pd.DataFrame) -> str:
    output_path = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
    df.to_csv(output_path, index=False)
    return output_path


# ---------- CORRELATION REPORT ----------
def get_correlation_report(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.shape[1] < 2:
        report = {
            "numeric_columns": numeric_df.columns.tolist(),
            "top_pairs": [],
            "summary": "Not enough numeric columns for correlation analysis."
        }

        with open(os.path.join(OUTPUT_DIR, "correlation_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    corr_matrix = numeric_df.corr().round(4)
    cols = corr_matrix.columns.tolist()

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            corr_val = float(corr_matrix.loc[c1, c2])
            abs_strength = abs(corr_val)

            if abs_strength >= 0.7:
                interpretation = "Strong relationship"
            elif abs_strength >= 0.4:
                interpretation = "Moderate relationship"
            else:
                interpretation = "Weak relationship"

            pairs.append({
                "col1": c1,
                "col2": c2,
                "pearson_corr": round(corr_val, 4),
                "spearman_corr": round(corr_val, 4),
                "interpretation": interpretation,
                "abs_strength": abs_strength
            })

    pairs = sorted(pairs, key=lambda x: x["abs_strength"], reverse=True)

    top_pairs = []
    for pair in pairs[:10]:
        top_pairs.append({
            "col1": pair["col1"],
            "col2": pair["col2"],
            "pearson_corr": pair["pearson_corr"],
            "spearman_corr": pair["spearman_corr"],
            "interpretation": pair["interpretation"]
        })

    report = {
        "numeric_columns": numeric_df.columns.tolist(),
        "top_pairs": top_pairs,
        "summary": f"Computed correlations across {len(numeric_df.columns)} numeric columns."
    }

    with open(os.path.join(OUTPUT_DIR, "correlation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ---------- SAVE GENERATED CODE ----------
def save_generated_code(code: str) -> str:
    output_path = os.path.join(OUTPUT_DIR, "generated_visual_code.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    return output_path