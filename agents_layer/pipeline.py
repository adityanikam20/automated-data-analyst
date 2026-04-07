import json
from pathlib import Path

from dotenv import load_dotenv

from agents_layer.tools import (
    load_dataset,
    get_basic_profile,
    clean_dataframe,
    save_cleaned_dataset,
    get_correlation_report,
    save_generated_code,
)
from agents_layer.llm_client import ask_llm

load_dotenv()


def build_visualization_prompt(profile: dict, cleaning_report: dict, corr_report: dict) -> str:
    return f"""
You are generating Python visualization code for a professional dashboard.

Dataset profile:
{json.dumps(profile, indent=2, default=str)}

Cleaning report:
{json.dumps(cleaning_report, indent=2, default=str)}

Correlation report:
{json.dumps(corr_report, indent=2, default=str)}

Requirements:
- Use pandas and plotly
- Read cleaned dataset from outputs/cleaned_data.csv
- Save charts to outputs/charts/
- Generate:
  1. missing values bar chart
  2. correlation heatmap
  3. scatter plot for strongest numeric pair
  4. histogram for one numeric column
- Return only valid Python code
"""


def get_fallback_visualization_code() -> str:
    return """import os
import pandas as pd
import plotly.express as px

os.makedirs("outputs/charts", exist_ok=True)

df = pd.read_csv("outputs/cleaned_data.csv")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Missing values chart
missing = df.isnull().sum().reset_index()
missing.columns = ["Column", "Missing"]
fig_missing = px.bar(missing, x="Column", y="Missing", title="Missing Values by Column")
fig_missing.write_html("outputs/charts/missing_values.html")

# Histogram
if numeric_cols:
    fig_hist = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
    fig_hist.write_html("outputs/charts/histogram.html")

# Correlation heatmap
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    fig_corr.write_html("outputs/charts/correlation_heatmap.html")

    corr_vals = corr.unstack().sort_values(ascending=False)
    corr_vals = corr_vals[corr_vals < 1]

    if not corr_vals.empty:
        x = corr_vals.index[0][0]
        y = corr_vals.index[0][1]
        fig_scatter = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
        fig_scatter.write_html("outputs/charts/top_scatter.html")
"""


def run_eda_pipeline(file_path: str) -> dict:
    file_path = str(Path(file_path))

    # 1. load dataset
    df = load_dataset(file_path)

    # 2. basic profile
    profile = get_basic_profile(df)

    # 3. clean dataset
    cleaned_df, cleaning_report = clean_dataframe(df)
    cleaned_file_path = save_cleaned_dataset(cleaned_df)

    # 4. correlation analysis
    correlation_report = get_correlation_report(cleaned_df)

    # 5. visualization code generation with fallback
    try:
        viz_input = build_visualization_prompt(profile, cleaning_report, correlation_report)
        visualization_code = ask_llm(viz_input)
    except Exception:
        visualization_code = get_fallback_visualization_code()

    generated_code_file = save_generated_code(visualization_code)

    return {
        "profile": profile,
        "cleaning_report": cleaning_report,
        "correlation_report": correlation_report,
        "cleaned_file": cleaned_file_path,
        "generated_code_file": generated_code_file,
    }