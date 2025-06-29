import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the simulated data
file_path = 'simulated_saas_data.csv'
df = pd.read_csv(file_path)

# Prepare dashboard figures
fig_arr_growth = px.scatter(
    df,
    x='arr_million',
    y='growth_rate_pct',
    color='acquisition_candidate',
    hover_data=['company_name', 'burn_rate_pct', 'churn_rate_pct'],
    title='ARR vs. YoY Growth (Acquisition Suitability)'
)

fig_burn_vs_runway = px.scatter(
    df,
    x='burn_rate_pct',
    y='runway_months',
    color='acquisition_candidate',
    hover_data=['company_name', 'arr_million'],
    title='Burn Rate vs. Runway'
)

# fig_margin_distribution = px.histogram(
#     df,
#     x='Gross Margin (%)',
#     color='Suitable for Acquisition',
#     nbins=20,
#     title='Distribution of Gross Margins'
# )

# fig_churn_vs_nrr = px.scatter(
#     df,
#     x='churn_rate_pct',
#     y='Net Revenue Retention (%)',
#     color='Suitable for Acquisition',
#     hover_data=['Company Name', 'ARR ($M)'],
#     title='Churn vs. NRR'
# )

fig_arr_box = px.box(
    df,
    x='acquisition_candidate',
    y='arr_million',
    title='ARR Distribution by Acquisition Suitability'
)

fig_arr_growth.write_html("arr_vs_growth.html")
fig_burn_vs_runway.write_html("burn_vs_runway.html")
fig_arr_box.write_html("arr_boxplot.html")

