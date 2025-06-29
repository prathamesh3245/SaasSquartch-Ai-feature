import streamlit as st
import pandas as pd
import plotly.express as px


df = pd.read_csv("simulated_leads.csv")

st.set_page_config(page_title="SaaSIntelDash", layout="wide")

st.title("📊 SaaS Lead Generation Intelligence Dashboard")
st.markdown("**Simulated dashboard to help Caprae & SaaSquatchLeads make smarter sales decisions**")


st.sidebar.header("🎯 Filter Leads")
industry = st.sidebar.multiselect("Select Industry", df['Industry'].unique())
country = st.sidebar.multiselect("Select Country", df['Country'].unique())

filtered_df = df.copy()
if industry:
    filtered_df = filtered_df[filtered_df['Industry'].isin(industry)]
if country:
    filtered_df = filtered_df[filtered_df['Country'].isin(country)]


st.subheader("🔢 Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Leads", len(filtered_df))
col2.metric("Avg Revenue ($M)", round(filtered_df["Revenue"].mean(), 2))
col3.metric("Avg Email Score", round(filtered_df["EmailScore"].mean(), 2))


st.subheader("💼 Revenue vs Company Size")
fig1 = px.scatter(filtered_df, x="Employees", y="Revenue", size="EmailScore",
                 color="Industry", hover_name="CompanyName",
                 title="Bigger Circles = Better Emails")
st.plotly_chart(fig1, use_container_width=True)


st.subheader("📌 Lead Sources")
fig2 = px.pie(filtered_df, names='Source', title="Distribution of Lead Sources")
st.plotly_chart(fig2, use_container_width=True)


if 'Country' in filtered_df.columns:
    st.subheader("🗺️ Leads by Country")
    geo_df = filtered_df.groupby("Country").size().reset_index(name="Count")
    fig3 = px.choropleth(geo_df, locations="Country", locationmode='country names',
                         color="Count", color_continuous_scale="Blues",
                         title="Lead Distribution Heatmap")
    st.plotly_chart(fig3, use_container_width=True)


st.subheader("🚦Lead Score Flag")
df['High Quality'] = df['EmailScore'].apply(lambda x: 'Yes' if x >= 80 else 'No')
st.dataframe(filtered_df[['CompanyName', 'Revenue', 'EmailScore']])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Drop rows with missing values (if any)
model_df = df.dropna()

# Encode 'High Quality' as target
df['High Quality'] = df['EmailScore'].apply(lambda x: 'Yes' if x >= 80 else 'No')

# ✅ Drop rows with missing Revenue or EmailScore only
model_df = df[['Revenue', 'EmailScore', 'High Quality']].dropna()

if model_df.empty:
    st.error("🚫 Not enough clean data (Revenue or EmailScore missing) to train the model.")
else:
    # ✅ Encode labels
    model_df['High Quality'] = model_df['High Quality'].map({'Yes': 1, 'No': 0})

    # Features and target
    features = ['Revenue', 'EmailScore']
    X = model_df[features]
    y = model_df['High Quality']

    # ✅ Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ✅ Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    st.success("✅ ML model trained successfully!")

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on filtered data
st.subheader("🤖 Lead Scoring Predictions")
if st.button("Score Leads"):
    filtered_X = filtered_df[features]
    predictions = clf.predict(filtered_X)
    prediction_labels = ['Yes' if p == 1 else 'No' for p in predictions]

    scored_df = filtered_df.copy()
    scored_df['Predicted High Quality'] = prediction_labels
    st.dataframe(scored_df[['CompanyName', 'Revenue', 'EmailScore', 'Predicted High Quality']])

import io


st.subheader("📤 Download Scored Leads")

if 'scored_df' in locals():
    to_download = scored_df[['CompanyName', 'Revenue', 'EmailScore', 'Predicted High Quality']]
    
    # Convert to Excel in memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        to_download.to_excel(writer, index=False, sheet_name='ScoredLeads')
    buffer.seek(0)

    st.download_button(
        label="📥 Download Excel File",
        data=buffer,
        file_name="scored_leads.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Please run lead scoring first.")
