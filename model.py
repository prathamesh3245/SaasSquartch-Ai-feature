import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Load the uploaded CSV
df = pd.read_csv("simulated_deals.csv")

# Basic cleanup: drop NA and select numerical features for similarity
features = ['Revenue', 'EmailScore', 'domain_age_years', 'num_funding_rounds', 'confidence_score']
df_clean = df.dropna(subset=features).copy()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# Calculate similarity matrix
similarity_matrix = cosine_similarity(X_scaled)

# Apply clustering for node color grouping
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_clean["cluster"] = clusters

# Create edge list from similarity matrix (threshold-based)
edges = []
threshold = 0.90  # high similarity
for i in range(similarity_matrix.shape[0]):
    for j in range(i + 1, similarity_matrix.shape[1]):
        if similarity_matrix[i, j] > threshold:
            edges.append((i, j, similarity_matrix[i, j]))

# Generate node positions using circular layout
theta = np.linspace(0, 2 * np.pi, len(df_clean))
x_pos = np.cos(theta)
y_pos = np.sin(theta)

# Plotly network graph
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=1, color='gray'),
    hoverinfo='none',
    mode='lines'
)

for edge in edges:
    x0, y0 = x_pos[edge[0]], y_pos[edge[0]]
    x1, y1 = x_pos[edge[1]], y_pos[edge[1]]
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)

# Node trace
node_trace = go.Scatter(
    x=x_pos,
    y=y_pos,
    mode='markers+text',
    text=df_clean['CompanyName'],
    textposition='bottom center',
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        size=15,
        color=df_clean['cluster'],
        colorbar=dict(
            thickness=15,
            title='Cluster',
            xanchor='left',
            titleside='right'
        ),
        line_width=2
    ),
    hovertext=[
        f"{row['CompanyName']}<br>Revenue: {row['Revenue']}<br>Confidence: {row['confidence_score']}" 
        for _, row in df_clean.iterrows()
    ],
    hoverinfo='text'
)

# Final figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Influence Network of Investment-Ready Leads',
                    titlefont_size=20,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

fig.show()
