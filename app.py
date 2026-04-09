import textwrap
import tempfile
import json
import os
from io import BytesIO
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.io as pio

pio.templates.default = "plotly_dark"
COLOR_THEME = px.colors.qualitative.Vivid

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from agents_layer.pipeline import run_eda_pipeline

@st.cache_data
def load_data(file_path):
    file_path = str(file_path)

    if file_path.lower().endswith(".csv"):
        try:
            return pd.read_csv(file_path, encoding="utf-8")
        except:
            return pd.read_csv(file_path, encoding="latin-1")
    elif file_path.lower().endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and XLSX files are supported.")

@st.cache_data
def run_pipeline_cached(file_bytes, file_name):
    suffix = "." + file_name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        file_path = tmp.name

    return run_eda_pipeline(file_path)

@st.cache_data
def get_summary_cached(df, numeric_cols, categorical_cols):
    return generate_summary_paragraph(df, numeric_cols, categorical_cols)

@st.cache_data
def get_corr_cached(df):
    return get_filtered_correlation_report(df)


def get_chart_recommendations(df, numeric_cols, categorical_cols):
    recommendations = []

    if numeric_cols:
        recommendations.append({
            "type": "histogram",
            "title": f"Distribution of {numeric_cols[0]}",
            "column": numeric_cols[0],
            "why": f"{numeric_cols[0]} is numeric, so a histogram is useful to understand spread and concentration."
        })

    if len(numeric_cols) >= 2:
        recommendations.append({
            "type": "scatter",
            "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "why": f"{numeric_cols[0]} and {numeric_cols[1]} are numeric, so a scatter plot is useful to inspect relationship."
        })

    if categorical_cols:
        recommendations.append({
            "type": "bar",
            "title": f"{categorical_cols[0]} distribution",
            "column": categorical_cols[0],
            "why": f"{categorical_cols[0]} is categorical, so a bar chart is best for comparing counts."
        })

        if df[categorical_cols[0]].nunique(dropna=True) <= 8:
            recommendations.append({
                "type": "pie",
                "title": f"{categorical_cols[0]} composition",
                "column": categorical_cols[0],
                "why": f"{categorical_cols[0]} has a small number of categories, so a pie chart is still readable."
            })

    return recommendations


def generate_summary_paragraph(df, numeric_cols, categorical_cols):
    parts = []

    parts.append(f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    if numeric_cols:
        first_num = numeric_cols[0]
        parts.append(
            f"The variable {first_num} has an average value of {round(df[first_num].mean(), 2)}, "
            f"with values ranging from {round(df[first_num].min(), 2)} to {round(df[first_num].max(), 2)}."
        )

    if categorical_cols:
        first_cat = categorical_cols[0]
        mode_val = df[first_cat].mode(dropna=True)
        if not mode_val.empty:
            parts.append(f"The most common category in {first_cat} is '{mode_val.iloc[0]}'.")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        corr_vals = corr.unstack().sort_values(ascending=False)
        corr_vals = corr_vals[corr_vals < 1]
        if not corr_vals.empty:
            top_corr = corr_vals.iloc[0]
            x = corr_vals.index[0][0]
            y = corr_vals.index[0][1]
            parts.append(
                f"The strongest numeric relationship is between {x} and {y}, with a correlation of {round(top_corr, 2)}."
            )

    total_missing = int(df.isnull().sum().sum())
    if total_missing == 0:
        parts.append("No missing values remain in the cleaned dataset.")
    else:
        parts.append(f"There are {total_missing} missing values remaining after cleaning.")

    total_duplicates = int(df.duplicated().sum())
    if total_duplicates == 0:
        parts.append("No duplicate rows remain after cleaning.")
    else:
        parts.append(f"There are {total_duplicates} duplicate rows remaining after cleaning.")

    return " ".join(parts)


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df.copy()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    categorical_cols = filtered_df.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

    for col in categorical_cols[:3]:
        unique_vals = filtered_df[col].dropna().unique().tolist()
        unique_vals = sorted([str(v) for v in unique_vals])
        if 1 < len(unique_vals) <= 25:
            selected_vals = st.sidebar.multiselect(
                f"Filter {col}",
                options=unique_vals,
                default=unique_vals,
                key=f"filter_{col}"
            )
            if selected_vals:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_vals)]

    for col in numeric_cols[:3]:
        col_series = filtered_df[col].dropna()
        if not col_series.empty:
            min_val = float(col_series.min())
            max_val = float(col_series.max())

            if min_val != max_val:
                selected_range = st.sidebar.slider(
                    f"Range for {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"range_{col}"
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= selected_range[0]) &
                    (filtered_df[col] <= selected_range[1])
                ]

    return filtered_df


def get_filtered_correlation_report(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        return {
            "numeric_columns": numeric_cols,
            "top_pairs": [],
            "summary": "Not enough numeric columns for correlation analysis."
        }

    corr = df[numeric_cols].corr().round(4)
    pairs = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]
            val = float(corr.loc[col1, col2])
            abs_val = abs(val)

            if abs_val >= 0.7:
                interpretation = "Strong relationship"
            elif abs_val >= 0.4:
                interpretation = "Moderate relationship"
            else:
                interpretation = "Weak relationship"

            pairs.append({
                "col1": col1,
                "col2": col2,
                "pearson_corr": round(val, 4),
                "spearman_corr": round(val, 4),
                "interpretation": interpretation,
                "abs_strength": abs_val
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

    return {
        "numeric_columns": numeric_cols,
        "top_pairs": top_pairs,
        "summary": f"Computed correlations across {len(numeric_cols)} numeric columns on filtered data."
    }

def generate_pdf_report(
    df,
    summary_paragraph,
    cleaning_report,
    correlation_report,
    numeric_cols,
    categorical_cols,
    source_filename="uploaded_file",
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
        spaceBefore=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.black,
        wordWrap="CJK",
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.black,
        wordWrap="CJK",
    )

    header_bg = colors.HexColor("#dbeafe")
    accent_bg = colors.HexColor("#eff6ff")
    border_color = colors.HexColor("#94a3b8")
    zebra_bg = colors.HexColor("#f8fafc")
    title_color = colors.HexColor("#1d4ed8")

    elements = []

    export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Title
    elements.append(Paragraph("Automated Data Analyst", title_style))
    elements.append(Paragraph("Premium Filtered Data Report", subtitle_style))

    # Metadata cards
    meta_data = [
        [
            Paragraph("<b>Source File</b>", body_style),
            Paragraph(str(source_filename), body_style),
            Paragraph("<b>Exported At</b>", body_style),
            Paragraph(export_time, body_style),
        ],
        [
            Paragraph("<b>Rows</b>", body_style),
            Paragraph(str(df.shape[0]), body_style),
            Paragraph("<b>Columns</b>", body_style),
            Paragraph(str(df.shape[1]), body_style),
        ],
        [
            Paragraph("<b>Numeric Columns</b>", body_style),
            Paragraph(", ".join(numeric_cols) if numeric_cols else "None", body_style),
            Paragraph("<b>Categorical Columns</b>", body_style),
            Paragraph(", ".join(categorical_cols) if categorical_cols else "None", body_style),
        ],
    ]

    meta_table = Table(meta_data, colWidths=[95, 145, 105, 145])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), accent_bg),
        ("BOX", (0, 0), (-1, -1), 0.7, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # Dataset summary
    elements.append(Paragraph("Dataset Summary", heading_style))
    dataset_summary = [
        ["Metric", "Value"],
        ["Rows", str(df.shape[0])],
        ["Columns", str(df.shape[1])],
        ["Numeric Columns", Paragraph(", ".join(numeric_cols) if numeric_cols else "None", body_style)],
        ["Categorical Columns", Paragraph(", ".join(categorical_cols) if categorical_cols else "None", body_style)],
    ]

    dataset_table = Table(dataset_summary, colWidths=[155, 345])
    dataset_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), title_color),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.7, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra_bg]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(dataset_table)
    elements.append(Spacer(1, 14))

    # Cleaning report
    elements.append(Paragraph("Cleaning Report", heading_style))
    cleaning_data = [
        ["Metric", "Value"],
        ["Duplicates Removed", str(cleaning_report.get("duplicates_removed", 0))],
        ["Summary", Paragraph(str(cleaning_report.get("summary", "N/A")), body_style)],
    ]

    cleaning_table = Table(cleaning_data, colWidths=[155, 345])
    cleaning_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), title_color),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.7, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra_bg]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(cleaning_table)
    elements.append(Spacer(1, 14))

    # Correlation summary
    elements.append(Paragraph("Correlation Report", heading_style))
    corr_summary_data = [
        ["Metric", "Value"],
        ["Summary", Paragraph(str(correlation_report.get("summary", "N/A")), body_style)],
    ]

    corr_summary_table = Table(corr_summary_data, colWidths=[155, 345])
    corr_summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), title_color),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.7, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra_bg]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(corr_summary_table)
    elements.append(Spacer(1, 10))

    top_pairs = correlation_report.get("top_pairs", [])
    if top_pairs:
        corr_rows = [["Column 1", "Column 2", "Pearson", "Spearman", "Interpretation"]]
        for pair in top_pairs[:5]:
            pearson_val = float(pair["pearson_corr"])
            interpretation = str(pair.get("interpretation", ""))
            if abs(pearson_val) >= 0.7:
                interpretation = f"★ {interpretation}"

            corr_rows.append([
                Paragraph(str(pair["col1"]), small_style),
                Paragraph(str(pair["col2"]), small_style),
                Paragraph(f"{pearson_val:.3f}", small_style),
                Paragraph(f"{float(pair['spearman_corr']):.3f}", small_style),
                Paragraph(interpretation, small_style),
            ])

        corr_table = Table(corr_rows, colWidths=[95, 95, 70, 70, 170])
        corr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), title_color),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOX", (0, 0), (-1, -1), 0.7, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra_bg]),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(corr_table)
    else:
        elements.append(Paragraph("No strong correlations found.", body_style))

    elements.append(Spacer(1, 14))

    # AI summary
    elements.append(Paragraph("AI Summary", heading_style))
    ai_summary_table = Table(
        [["Summary"], [Paragraph(summary_paragraph, body_style)]],
        colWidths=[500],
    )
    ai_summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (0, 0), title_color),
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.7, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (0, 1), colors.white),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(ai_summary_table)
    elements.append(Spacer(1, 14))

    # Data preview
    elements.append(Paragraph("Preview of Filtered Data", heading_style))
    preview_df = df.head(10).copy()

    preview_rows = [[Paragraph(str(col), small_style) for col in preview_df.columns]]
    for _, row in preview_df.iterrows():
        preview_rows.append([
            Paragraph(str(val) if len(str(val)) <= 22 else str(val)[:19] + "...", small_style)
            for val in row.tolist()
        ])

    col_count = len(preview_df.columns)
    usable_width = 500
    col_width = usable_width / max(col_count, 1)
    preview_table = Table(preview_rows, colWidths=[col_width] * col_count)
    preview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), title_color),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.7, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra_bg]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(preview_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer
# ---------- STYLES ----------
st.markdown("""
<style>
.sidebar-preview-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 10px;
    margin-top: 8px;
    margin-bottom: 10px;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #243041;
}
.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 30px;
    border-radius: 22px;
    border: 1px solid #334155;
    margin-bottom: 20px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
}
.kpi-card {
    background: #111827;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #374151;
    text-align: center;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.12);
}
.kpi-label {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.kpi-value {
    color: white;
    font-size: 30px;
    font-weight: 700;
}
.section-card {
    background: #111827;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #334155;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.10);
}
.download-card {
    background: #0f172a;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #334155;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.small-muted {
    color: #94a3b8;
    font-size: 14px;
}
.insight-card {
    background: #0f172a;
    padding: 14px 16px;
    border-radius: 14px;
    border: 1px solid #334155;
    margin-bottom: 10px;
}
.recommend-card {
    background: #0f172a;
    padding: 14px 16px;
    border-radius: 14px;
    border: 1px solid #334155;
    margin-bottom: 12px;
}
.summary-card {
    background: #0f172a;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #334155;
    line-height: 1.8;
    color: #e5e7eb;
}
.subtle-divider {
    height: 1px;
    background: #243041;
    margin: 10px 0 18px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("📂 Automated Data Analyst")
st.sidebar.caption("Upload a CSV or XLSX file")
uploaded_file = st.sidebar.file_uploader("Choose file", type=["csv", "xlsx"])

# ---------- SIDEBAR FILE PREVIEW ----------
if uploaded_file is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("File Preview")
    st.sidebar.markdown('<div class="sidebar-preview-box">', unsafe_allow_html=True)

    try:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext == "csv":
            try:
                preview_df = pd.read_csv(uploaded_file, encoding="utf-8")
            except:
                preview_df = pd.read_csv(uploaded_file, encoding="latin-1")
        elif file_ext == "xlsx":
            preview_df = pd.read_excel(uploaded_file)
        else:
            preview_df = None

        uploaded_file.seek(0)

        if preview_df is not None:
            st.sidebar.caption(f"Rows: {preview_df.shape[0]} | Columns: {preview_df.shape[1]}")
            st.sidebar.dataframe(
                preview_df.head(5),
                use_container_width=True,
                height=180
            )
    except Exception:
        st.sidebar.warning("Preview not available for this file.")

    st.sidebar.markdown("</div>", unsafe_allow_html=True)
# ---------- LANDING ----------
if not uploaded_file:
    st.markdown("""
    <div class="hero-card">
        <h1 style="color:white; margin-bottom:8px;">📊 Automated Data Analyst</h1>
        <p style="color:#cbd5e1; font-size:18px; margin-bottom:0;">
            Upload your dataset and get automated cleaning, correlation analysis, smart chart recommendations,
            downloadable reports, and visualization code.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="section-card">
            <h3>🧹 Data Cleaning</h3>
            <p class="small-muted">Standardizes data, removes duplicates, and prepares the dataset for analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-card">
            <h3>📈 Smart Visualization</h3>
            <p class="small-muted">Generates histograms, box plots, scatter plots, bar charts, pie charts, and correlations.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="section-card">
            <h3>⬇️ Downloadable Outputs</h3>
            <p class="small-muted">Export the cleaned dataset, structured EDA report, and generated visualization code.</p>
        </div>
        """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    for col, label in zip([k1, k2, k3, k4], ["Rows", "Columns", "Numeric Columns", "Duplicates Removed"]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">—</div>
            </div>
            """, unsafe_allow_html=True)

# ---------- APP ----------
else:
    try:
        with st.spinner("Running EDA pipeline..."):
            results = run_pipeline_cached(
                uploaded_file.getvalue(),
                uploaded_file.name
            )
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
        st.stop()

    raw_df = load_data(results["cleaned_file"])

    if raw_df.empty:
        st.warning("⚠️ Uploaded file has no data.")
        st.stop()

    df = apply_sidebar_filters(raw_df)

    if df.empty:
        st.warning("⚠️ No data after applying filters.")
        st.stop()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    summary_paragraph = get_summary_cached(df, numeric_cols, categorical_cols)
    filtered_correlation_report = get_corr_cached(df)

    #DATA QUALITY SCORE
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    duplicate_pct = df.duplicated().sum() / df.shape[0] * 100

    score = 100 - (missing_pct * 0.5 + duplicate_pct * 0.5)
    score = round(max(score, 0), 2)
    if score >= 90:
        score_color = "#22c55e"
    elif score >= 70:
        score_color = "#f59e0b"
    else:
        score_color = "#ef4444"

    if score >= 90:
        score_label = "Good"
        score_message = "Dataset quality is strong. Missing values and duplicates are low."
    elif score >= 70:
        score_label = "Moderate"
        score_message = "Dataset is usable, but some cleaning issues are still present."
    else:
        score_label = "Poor"
        score_message = "Dataset needs more cleaning before reliable analysis."
    # ---------- HERO ----------
    st.markdown(f"""
    <div class="hero-card">
        <h1 style="color:white; margin-bottom:8px;">📊 Automated Data Analyst</h1>
        <p style="color:#cbd5e1; font-size:17px; margin-bottom:6px;">
            Analysis completed for <b>{uploaded_file.name}</b>
        </p>
        <p class="small-muted" style="margin-bottom:0;">
            Automated cleaning, profiling, correlations, chart recommendations, insights, and downloads.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- KPI ----------
    c1, c2, c3, c4 , c5= st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Filtered Rows</div>
            <div class="kpi-value">{df.shape[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Columns</div>
            <div class="kpi-value">{df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Numeric Columns</div>
            <div class="kpi-value">{len(numeric_cols)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Duplicates Removed</div>
            <div class="kpi-value">{results["cleaning_report"]["duplicates_removed"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Data Quality</div>
            <div class="kpi-value">{score}</div>
            <div style="
                display:inline-block;
                margin-top:8px;
                padding:4px 10px;
                border-radius:999px;
                background:{score_color};
                color:white;
                font-size:12px;
                font-weight:600;">
                {score_label}
            </div>
            <div style="background:#374151; border-radius:8px; height:10px; margin-top:12px;">
                <div style="
                    width:{score}%;
                    background:{score_color};
                    height:10px;
                    border-radius:8px;">
                </div>
            </div>
            <div style="margin-top:10px; font-size:12px; color:#94a3b8;">
                {score_message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Correlation", "Charts", "Insights", "Downloads"]
    )

    # ---------- OVERVIEW ----------
    with tab1:
        left, right = st.columns([1.35, 1])

        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Dataset Preview")
            st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Missing Values")
            st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Column", "Missing"]
            fig_missing = px.bar(
                missing,
                x="Column",
                y="Missing",
                title="Missing Values by Column",
                color="Column",
                color_discrete_sequence=COLOR_THEME
            )
            fig_missing.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_missing, use_container_width=True, key="missing_chart")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Quick Summary")
        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
        st.write(f"Filtered rows: {df.shape[0]}")
        st.write(f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}")
        st.write(f"Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}")
        st.markdown('</div>', unsafe_allow_html=True)

        h1, h2, h3 = st.columns(3)
        with h1:
            if numeric_cols:
                top_numeric = numeric_cols[0]
                st.markdown(f"""
                <div class="section-card">
                    <h4>Average {top_numeric}</h4>
                    <p style="font-size:24px; font-weight:700; color:white; margin:0;">{round(df[top_numeric].mean(), 2)}</p>
                </div>
                """, unsafe_allow_html=True)

        with h2:
            if categorical_cols:
                top_cat = categorical_cols[0]
                top_value = df[top_cat].mode(dropna=True)
                if not top_value.empty:
                    st.markdown(f"""
                    <div class="section-card">
                        <h4>Most Common {top_cat}</h4>
                        <p style="font-size:24px; font-weight:700; color:white; margin:0;">{top_value.iloc[0]}</p>
                    </div>
                    """, unsafe_allow_html=True)

        with h3:
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                corr_vals = corr.unstack().sort_values(ascending=False)
                corr_vals = corr_vals[corr_vals < 1]
                if not corr_vals.empty:
                    top_corr = corr_vals.iloc[0]
                    x = corr_vals.index[0][0]
                    y = corr_vals.index[0][1]
                    st.markdown(f"""
                    <div class="section-card">
                        <h4>Top Correlation</h4>
                        <p style="font-size:16px; font-weight:700; color:white; margin:0;">{x} ↔ {y}</p>
                        <p class="small-muted" style="margin-top:6px;">{round(top_corr, 2)}</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("AI-Style Summary")
        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card">{summary_paragraph}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- CORRELATION ----------
    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Correlation Analysis")
        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                title="Correlation Heatmap",
                color_continuous_scale="turbo"
            )
            fig_corr.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_heatmap")

            corr_vals = corr.unstack().sort_values(ascending=False)
            corr_vals = corr_vals[corr_vals < 1]

            if not corr_vals.empty:
                top = corr_vals.iloc[0]
                st.success(f"Strongest correlation: {round(top, 2)}")

                x = corr_vals.index[0][0]
                y = corr_vals.index[0][1]
                fig_top_scatter = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
                fig_top_scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_top_scatter, use_container_width=True, key="top_scatter")
        else:
            st.info("Not enough numeric columns for correlation analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- CHARTS ----------
    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("AI Chart Recommendations")
        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        recommendations = get_chart_recommendations(df, numeric_cols, categorical_cols)

        for i, rec in enumerate(recommendations):
            st.markdown(
                f'<div class="recommend-card"><b>{i+1}. {rec["title"]}</b><br><span class="small-muted">{rec["why"]}</span></div>',
                unsafe_allow_html=True
            )

            if rec["type"] == "histogram":
                fig = px.histogram(
                    df,
                    x=rec["column"],
                    title=rec["title"],
                    color_discrete_sequence=COLOR_THEME
                )
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True, key=f"rec_hist_{i}")

            elif rec["type"] == "scatter":
                fig = px.scatter(
                    df,
                    x=rec["x"],
                    y=rec["y"],
                    title=rec["title"],
                    color=df[rec["x"]],
                    color_continuous_scale="turbo"
                )
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True, key=f"rec_scatter_{i}")

            elif rec["type"] == "bar":
                count_df = df[rec["column"]].value_counts().reset_index()
                count_df.columns = [rec["column"], "count"]
                fig = px.bar(
                    count_df,
                    x=rec["column"],
                    y="count",
                    title=rec["title"],
                    color=rec["column"],
                    color_discrete_sequence=COLOR_THEME
                )
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True, key=f"rec_bar_{i}")

            elif rec["type"] == "pie":
                pie_df = df[rec["column"]].value_counts().reset_index()
                pie_df.columns = [rec["column"], "count"]
                fig = px.pie(
                    pie_df,
                    names=rec["column"],
                    values="count",
                    title=rec["title"],
                    color_discrete_sequence=COLOR_THEME
                )
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True, key=f"rec_pie_{i}")

        st.markdown('</div>', unsafe_allow_html=True)

        if numeric_cols:
            a, b = st.columns(2)

            with a:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("Histogram")
                st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
                hist_col = st.selectbox("Histogram Column", numeric_cols, key="hist_select")
                fig_hist = px.histogram(
                    df,
                    x=hist_col,
                    title=f"Distribution of {hist_col}",
                    color_discrete_sequence=COLOR_THEME
                )
                fig_hist.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{hist_col}")
                st.markdown('</div>', unsafe_allow_html=True)

            with b:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("Box Plot")
                st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
                box_col = st.selectbox("Box Column", numeric_cols, key="box_select")
                fig_box = px.box(
                    df,
                    y=box_col,
                    title=f"Box Plot of {box_col}",
                    color_discrete_sequence=COLOR_THEME
                )
                fig_box.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_box, use_container_width=True, key=f"box_{box_col}")
                st.markdown('</div>', unsafe_allow_html=True)

        if len(numeric_cols) >= 2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Custom Scatter Plot")
            st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            with s1:
                x_col = st.selectbox("X Axis", numeric_cols, key="scatter_x")
            with s2:
                y_col = st.selectbox("Y Axis", numeric_cols, key="scatter_y")
            fig_scatter = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}",
                color=df[x_col],
                color_continuous_scale="turbo"
            )
            fig_scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{x_col}_{y_col}")
            st.markdown('</div>', unsafe_allow_html=True)

        if categorical_cols:
            a, b = st.columns(2)

            with a:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("Bar Chart")
                st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
                cat_col = st.selectbox("Category Column", categorical_cols, key="bar_select")
                count_df = df[cat_col].value_counts(dropna=False).reset_index()
                count_df.columns = [cat_col, "count"]
                fig_bar = px.bar(
                    count_df,
                    x=cat_col,
                    y="count",
                    title=f"Count of {cat_col}",
                    color=cat_col,
                    color_discrete_sequence=COLOR_THEME
                )
                fig_bar.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{cat_col}")
                st.markdown('</div>', unsafe_allow_html=True)

            with b:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("Pie Chart")
                st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
                pie_col = st.selectbox("Pie Column", categorical_cols, key="pie_select")
                pie_df = df[pie_col].value_counts(dropna=False).reset_index()
                pie_df.columns = [pie_col, "count"]
                fig_pie = px.pie(
                    pie_df,
                    names=pie_col,
                    values="count",
                    title=f"Composition of {pie_col}",
                    color_discrete_sequence=COLOR_THEME
                )
                fig_pie.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{pie_col}")
                st.markdown('</div>', unsafe_allow_html=True)

    # ---------- INSIGHTS ----------
    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Auto Insights")
        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        insights = []

        total_missing = int(df.isnull().sum().sum())
        total_duplicates = int(df.duplicated().sum())

        if total_missing == 0:
            insights.append("No missing values remain in the cleaned dataset.")
        else:
            insights.append(f"{total_missing} missing values are still present in the dataset.")

        if total_duplicates == 0:
            insights.append("No duplicate rows remain after cleaning.")
        else:
            insights.append(f"{total_duplicates} duplicate rows are still present.")

        if numeric_cols:
            for col in numeric_cols[:3]:
                insights.append(
                    f"{col}: average = {round(df[col].mean(), 2)}, minimum = {round(df[col].min(), 2)}, maximum = {round(df[col].max(), 2)}."
                )

        if categorical_cols:
            for col in categorical_cols[:2]:
                top_value = df[col].mode(dropna=True)
                if not top_value.empty:
                    insights.append(f"Most frequent value in {col} is '{top_value.iloc[0]}'.")

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            corr_vals = corr.unstack().sort_values(ascending=False)
            corr_vals = corr_vals[corr_vals < 1]

            if not corr_vals.empty:
                top_corr = corr_vals.iloc[0]
                x = corr_vals.index[0][0]
                y = corr_vals.index[0][1]
                insights.append(
                    f"Strongest detected relationship is between {x} and {y} with correlation {round(top_corr, 2)}."
                )

        for idx, item in enumerate(insights, start=1):
            st.markdown(f'<div class="insight-card"><b>{idx}.</b> {item}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- DOWNLOADS ----------
    with tab5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Download Outputs")
        st.markdown(
            '<div class="small-muted">Export filtered data, filtered report, PDF report, and generated code.</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.markdown("*Filtered Dataset*")
            st.markdown(
                '<div class="small-muted">Download the currently filtered dataset as CSV.</div>',
                unsafe_allow_html=True
            )
            st.caption("Preview")
            st.dataframe(df.head(3), use_container_width=True, height=140)

            filtered_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Filtered CSV",
                data=filtered_csv,
                file_name="filtered_data.csv",
                mime="text/csv",
                key="download_filtered_csv"
            )

        with d2:
            st.markdown("*Filtered Report*")
            st.markdown(
                '<div class="small-muted">Download summary and metadata for the currently filtered dataset.</div>',
                unsafe_allow_html=True
            )

            filtered_report = {
                "filtered_shape": [df.shape[0], df.shape[1]],
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "summary_paragraph": summary_paragraph,
                "cleaning_report": results["cleaning_report"],
                "correlation_report": filtered_correlation_report,
            }

            st.caption("Preview")
            report_preview = {
                "filtered_shape": filtered_report["filtered_shape"],
                "numeric_columns": filtered_report["numeric_columns"][:3],
                "categorical_columns": filtered_report["categorical_columns"][:3],
            }
            st.text_area(
                "report_preview",
                json.dumps(report_preview, indent=2),
                height=140,
                disabled=True,
                label_visibility="collapsed"
            )

            st.download_button(
                label="Download Report JSON",
                data=json.dumps(filtered_report, indent=4),
                file_name="filtered_eda_report.json",
                mime="application/json",
                key="download_filtered_json"
            )

        with d3:
            st.markdown("*PDF Report*")
            st.markdown(
                '<div class="small-muted">Download a PDF summary of the currently filtered dataset.</div>',
                unsafe_allow_html=True
            )

            st.caption("Preview")
            pdf_preview_text = f"""Rows: {df.shape[0]}
        Columns: {df.shape[1]}
        Top numeric columns: {", ".join(numeric_cols[:3]) if numeric_cols else "None"}
        Top categorical columns: {", ".join(categorical_cols[:3]) if categorical_cols else "None"}

        Summary:
        {summary_paragraph[:180]}"""

            st.text_area(
                "pdf_preview",
                pdf_preview_text,
                height=140,
                disabled=True,
                label_visibility="collapsed"
            )

            # ✅ BUTTONS INSIDE d3
            if st.button("Prepare PDF", key="prepare_pdf_btn"):
                pdf_buffer = generate_pdf_report(
                    df,
                    summary_paragraph,
                    results["cleaning_report"],
                    filtered_correlation_report,
                    numeric_cols,
                    categorical_cols,
                    source_filename=uploaded_file.name,
                )

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name="filtered_eda_report.pdf",
                    mime="application/pdf",
                    key="download_pdf_report"
                )

        with d4:
            st.markdown("*Visualization Code*")
            st.markdown(
                '<div class="small-muted">Download generated Python visualization code.</div>',
                unsafe_allow_html=True
            )

            if os.path.exists(results["generated_code_file"]):
                with open(results["generated_code_file"], "r", encoding="utf-8") as f:
                    code_text = f.read()

                st.caption("Preview")
                preview_lines = "\n".join(code_text.splitlines()[:10])
                st.text_area(
                    "code_preview",
                    preview_lines,
                    height=140,
                    disabled=True,
                    label_visibility="collapsed"
                )

                st.download_button(
                    label="Download Python Code",
                    data=code_text,
                    file_name="generated_visual_code.py",
                    mime="text/x-python",
                    key="download_python_code"
                )
            else:
                st.info("Visualization code file not found.")

        st.markdown('</div>', unsafe_allow_html=True)