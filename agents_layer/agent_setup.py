from agents import Agent
from agents_layer.schemas import CleaningReport, CorrelationReport, VizCodeReport


cleaner_agent = Agent(
    name="Cleaner Agent",
    instructions=(
        "You are a professional data cleaning analyst. "
        "You receive dataset profile information and cleaning results. "
        "Return a concise and professional structured report. "
        "Focus on duplicates removed, column cleanup, data type fixes, and missing-value handling. "
        "Keep summary clear and business-friendly."
    ),
    output_type=CleaningReport,
)

correlation_agent = Agent(
    name="Correlation Agent",
    instructions=(
        "You are a statistical analysis expert. "
        "You receive precomputed correlation findings for numeric columns. "
        "Return a structured professional summary of the strongest correlations. "
        "Never claim causation. "
        "Use short, accurate interpretations like strong, moderate, or weak relationship."
    ),
    output_type=CorrelationReport,
)

visualization_agent = Agent(
    name="Visualization Agent",
    instructions=(
        "You are a senior Python data visualization engineer. "
        "Generate clean, professional Python code using plotly.express and plotly.graph_objects. "
        "The code should assume the cleaned dataset exists at outputs/cleaned_data.csv. "
        "Generate code for: "
        "1) missing values bar chart, "
        "2) correlation heatmap, "
        "3) top correlation scatter plot, "
        "4) one histogram for a numeric column. "
        "Return only valid Python code inside the python_code field. "
        "The code should save charts to outputs/charts/. "
        "Also provide a short summary and list of generated filenames."
    ),
    output_type=VizCodeReport,
)