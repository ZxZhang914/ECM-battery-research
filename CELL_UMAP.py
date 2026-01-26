import pandas as pd
import umap
import plotly.express as px
from pathlib import Path

# Load data
data_path = Path("fulldf_global_median.csv")
if not data_path.exists():
    raise FileNotFoundError("Missing fulldf_global_median.csv.")
    
df = pd.read_csv(data_path, index_col=0)
# df = df[df['Temp'] == 25]

# UMAP on ECM features
features = ['R0', 'R1', 'R2', 'R3']
X = df[features].values

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)

df['UMAP1'] = embedding[:, 0]
df['UMAP2'] = embedding[:, 1]

# Interactive scatter plot
fig = px.scatter(
    df,
    x='UMAP1',
    y='UMAP2',
    color='CELL',  # color for each battery/cell
    color_discrete_sequence=px.colors.qualitative.Plotly,  # nice distinct palette
    hover_data={
        'CELL': True,
        'SOH': ':.3f',
        'SOC': ':.3f',
        'R0': ':.5f',
        'R1': ':.5f',
        'R2': ':.5f',
        'R3': ':.5f',
        'UMAP1': False,
        'UMAP2': False
    },
)

fig.update_traces(marker=dict(
    size=8,
    opacity=0.6 + 0.4 * (df['SOC'] / df['SOC'].max()),  # brighter at higher SOC
    line=dict(width=0.5, color='black')
))

fig.update_layout(
    title="UMAP of ECM Parameters [Features: R0,R1,R2,R3] (Colored by Cell)",
    xaxis_title="UMAP1",
    yaxis_title="UMAP2",
    template="plotly_white",
    legend_title_text="Cell"
)

fig.write_html("umap_allCELLs_plot.html")
print("Plot saved as umap_ecm_plot.html — open it in your browser.")
