import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt  # Needed for saving SHAP plot
import joblib
# ------------------------
# 1. Load and Prepare Data
# ------------------------
df = pd.read_csv("simulated_deals.csv")
df['deal_ready'] = df['deal_ready'].astype(int)

feature_names = [
    'norm_revenue', 'norm_funding', 'EmailScore', 'domain_age_years',
    'has_founder_linkedin', 'registry_verified', 'num_funding_rounds'
]

X = df[feature_names].values
y = df['deal_ready'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------
# 2. Normalize Features
# -------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_all = scaler.transform(X)

# -------------------
# 3. Convert to Torch Tensors
# -------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_all_tensor = torch.tensor(X_all, dtype=torch.float32)

# -------------------
# 4. Define NN Model
# -------------------
class LeadNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = LeadNN()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------
# 5. Train the Model
# -------------------
for epoch in range(300):
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----------------------------
# 6. Generate NN Predictions
# ----------------------------
nn_outputs = model(X_all_tensor).detach().numpy()
df['nn_confidence_score'] = nn_outputs
df.to_csv("simulated_deals_with_nn_score.csv", index=False)
print("Saved enhanced dataset with neural confidence score.") 

# -------------------
# 7. SHAP Explanation
# -------------------
# Use 100 samples for background
background = X_all_tensor[:100]
explainer = shap.DeepExplainer(model, background)



# Get SHAP values for binary classification
shap_values = explainer.shap_values(X_all_tensor)

# If output is a list (as is typical with classification), take the first class
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Remove the last dimension if it's 1
if shap_values.ndim == 3 and shap_values.shape[2] == 1:
    shap_values = shap_values.squeeze(-1)

# Convert to numpy
X_all_np = X_all_tensor.cpu().numpy()

# Final shape check
assert shap_values.shape == X_all_np.shape, f"Shape mismatch: {shap_values.shape} vs {X_all_np.shape}"


# Plot
shap.summary_plot(shap_values, X_all_np, feature_names=feature_names, show=False)



# SHAP values


# SHAP summary plot

plt.savefig("shap_summary.png")
plt.close()
print("SHAP summary saved as 'shap_summary.png'.")

# ---------------------------
# 8. SHAP Insights Generation
# ---------------------------
# Build DataFrame for SHAP
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df['nn_confidence_score'] = nn_outputs.flatten()
shap_df["lead_quality_score"] = (shap_df["nn_confidence_score"] * 100).round(2)


df["lead_quality_score"] = (df["nn_confidence_score"] * 100).round(2)



# Include original features for reference
for i, fname in enumerate(feature_names):
    shap_df[f'feat_{fname}'] = X_all[:, i]

# Insight function
def generate_insight(row, top_n=3):
    shap_scores = row[feature_names]
    top_features = shap_scores.abs().sort_values(ascending=False).head(top_n)

    insight = f"Lead with confidence score {row['lead_quality_score']:.2f} is influenced by:\n"
    for fname in top_features.index:
        effect = shap_scores[fname]
        direction = "positively" if effect > 0 else "negatively"
        insight += f"- {fname} contributes {direction} ({effect:+.2f})\n"
    return insight

# Show top 5 leads
top_leads = shap_df.sort_values(by="lead_quality_score", ascending=False).head(5)
# top_leads = df.sort_values(by="lead_quality_score", ascending=False).head(5)

for idx, row in top_leads.iterrows():
    print(f"\nInsight for Lead #{idx}")
    print(generate_insight(row))


df["deal_tier"] = pd.cut(df["lead_quality_score"],
                         bins=[0, 60, 80, 100],
                         labels=["Cold", "Warm", "Hot"])


mean_abs_shap = np.abs(shap_values).mean(axis=0)
impact_df = pd.DataFrame({
    "feature": feature_names,
    "avg_impact": mean_abs_shap
}).sort_values(by="avg_impact", ascending=False)
impact_df.to_csv("feature_impact.csv", index=False)



import json

# lead_insights = []
# for idx, row in top_leads.iterrows():
#     insight_text = generate_insight(row)
#     lead_insights.append({
#         "lead_quality_score": float(row["lead_quality_score"]),
#         "insight": insight_text
#     })

# with open("top_leads.json", "w") as f:
#     json.dump(lead_insights, f, indent=2)
top_leads_records = []

for idx, row in top_leads.iterrows():
    insight = generate_insight(row)
    top_leads_records.append({
        "id": int(idx),
        "confidence": float(row['nn_confidence_score']),
        "score": float(row['lead_quality_score']),
        "insight": insight,
        **{f: float(row[f'feat_{f}']) for f in feature_names}
    })

with open("top_leads.json", "w") as f:
    json.dump(top_leads_records, f, indent=2)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')

X_all_tensor_export = X_all_tensor  # Add this line at the end


# Save:
torch.save(model.state_dict(), "lead_nn.pt")

# Load:
model.load_state_dict(torch.load("lead_nn.pt"))
model.eval()

# ✅ Export for Flask access
trained_model = model
