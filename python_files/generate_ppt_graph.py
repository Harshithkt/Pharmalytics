import plotly.graph_objects as go
import pandas as pd
import os

# Data from our pipeline outputs
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
xgb_scores = [97.66, 97.69, 97.66, 97.66, 99.82] # Scaled ROC to 100% for parity
rf_scores = [93.75, 94.00, 93.75, 93.74, 98.95]

# Claude Theme Colors
color_xgb = '#D97757'  # Primary brand color
color_rf = '#5A5A5A'   # Muted secondary color
bg_color = '#FFFFFF'

fig = go.Figure()

# Add XGBoost Bars
fig.add_trace(go.Bar(
    name='XGBoost (Best Model)',
    x=metrics,
    y=xgb_scores,
    marker_color=color_xgb,
    text=[f"{val:.2f}%" for val in xgb_scores],
    textposition='auto',
    textfont=dict(color='white', size=14, family="Inter")
))

# Add Random Forest Bars
fig.add_trace(go.Bar(
    name='Random Forest',
    x=metrics,
    y=rf_scores,
    marker_color=color_rf,
    text=[f"{val:.2f}%" for val in rf_scores],
    textposition='auto',
    textfont=dict(color='white', size=14, family="Inter")
))

# Update Layout for a premium PPT look
fig.update_layout(
    title=dict(
        text="<b>AI Bottleneck Predictor: Model Performance Comparison (5-Fold CV)</b>",
        font=dict(size=24, color='#2F2F2F', family="Inter")
    ),
    barmode='group',
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(family="Inter", size=16, color='#2F2F2F'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=16)
    ),
    margin=dict(t=100, b=50, l=50, r=50),
    yaxis=dict(
        title="Score (%)",
        range=[90, 102],
        gridcolor='#E6DFD2',
        zerolinecolor='#E6DFD2',
        showline=True,
        linecolor='#E6DFD2',
        linewidth=2
    ),
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='#E6DFD2',
        linewidth=2
    ),
    width=1100,
    height=600
)

# Save high-res PNG for PowerPoint
output_path = "Model_Performance_Validation_PPT.png"
fig.write_image(output_path, scale=2)
print(f"Graph successfully generated and saved to: {os.path.abspath(output_path)}")
