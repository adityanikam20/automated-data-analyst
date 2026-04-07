```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Create output directory for charts
output_dir = 'outputs/charts'
os.makedirs(output_dir, exist_ok=True)

# Read cleaned dataset
df = pd.read_csv('outputs/cleaned_data.csv')

# 1. Missing values bar chart
missing_counts = df.isnull().sum()
fig1 = px.bar(
    x=missing_counts.index,
    y=missing_counts.values,
    labels={'x': 'Column', 'y': 'Missing Values Count'},
    title='Missing Values per Column',
    text_auto=True
)
fig1.update_layout(showlegend=False)
fig1.write_html(f'{output_dir}/missing_values_bar_chart.html')

# 2. Correlation heatmap
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr()
    fig2 = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title='Correlation Heatmap'
    )
else:
    # Create empty heatmap for single numeric column
    fig2 = go.Figure()
    fig2.add_annotation(
        text="Not enough numeric columns for correlation analysis",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    fig2.update_layout(title='Correlation Heatmap')

fig2.write_html(f'{output_dir}/correlation_heatmap.html')

# 3. Scatter plot for strongest numeric pair
if len(numeric_cols) >= 2:
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    # Mask diagonal to avoid self-correlation
    import numpy as np
    mask = np.eye(len(corr_matrix), dtype=bool)
    corr_matrix_masked = corr_matrix.mask(mask)
    
    # Find strongest correlation pair
    max_corr = corr_matrix_masked.max().max()
    max_pair = np.where(corr_matrix_masked == max_corr)
    col1 = numeric_cols[max_pair[0][0]]
    col2 = numeric_cols[max_pair[1][0]]
    
    fig3 = px.scatter(
        df,
        x=col1,
        y=col2,
        title=f'Scatter Plot: {col1} vs {col2} (r = {max_corr:.3f})',
        trendline='ols'
    )
else:
    fig3 = go.Figure()
    fig3.add_annotation(
        text="Not enough numeric columns for scatter plot",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    fig3.update_layout(title='Scatter Plot of Strongest Numeric Pair')

fig3.write_html(f'{output_dir}/scatter_plot_strongest_pair.html')

# 4. Histogram for one numeric column
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    fig4 = px.histogram(
        df,
        x=col,
        title=f'Histogram: {col}',
        nbins=min(10, len(df)),  # Adjust bins based on data size
        marginal='box'
    )
    fig4.update_layout(bargap=0.1)
else:
    fig4 = go.Figure()
    fig4.add_annotation(
        text="No numeric columns available for histogram",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    fig4.update_layout(title='Histogram of Numeric Column')

fig4.write_html(f'{output_dir}/histogram_numeric_column.html')
```