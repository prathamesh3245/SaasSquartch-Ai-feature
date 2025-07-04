from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import numpy as np
import torch
from model import trained_model as lead_model, feature_names, X_all_tensor as X_all_tensor_export

from model import feature_names
import joblib
from sklearn.preprocessing import StandardScaler
import csv
from pathlib import Path
import shap
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React or JS

@app.route('/', methods=['GET'])
def index():
    return render_template('index2.html')

@app.route('/top-leads', methods=['GET'])
def get_top_leads():
    try:
        with open('top_leads.json') as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/shap-summary', methods=['GET'])
def get_shap_plot():
    return send_from_directory('static', 'shap_summary.png')

# Optional: A POST endpoint to receive user input or lead data
@app.route("/submit-lead", methods=["POST"])
def submit_lead():
    data = request.json
    try:
        input_vector = np.array([[data[f] for f in feature_names]])
        input_scaled = scaler.transform(input_vector)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        confidence = lead_model(input_tensor).item()

        score = round(confidence * 100, 2)

        def categorize(score):
            if score < 60:
                return "Cold"
            elif score < 80:
                return "Warm"
            return "Hot"
        deal_tier = categorize(score)

        csv_path = Path("submitted_leads.csv")
        file_exists = csv_path.exists()

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(feature_names + ['confidence_score', 'deal_tier'])  # header
            writer.writerow([data[f] for f in feature_names] + [score, deal_tier])

        # background = lead_model.X_all_tensor[:100]  # reuse same background
        # explainer = shap.DeepExplainer(lead_model.model, background)
        background = X_all_tensor_export[:100]  # ✅ Corrected
        explainer = shap.DeepExplainer(lead_model.model, background)


        shap_values = explainer.shap_values(input_tensor)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values.squeeze()

        top_features = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]

        insight = f"🧠 Lead with confidence score {score:.2f} is influenced by:\n"
        for fname, val in top_features:
            direction = "positively" if val > 0 else "negatively"
            insight += f"- {fname} contributes {direction} ({val:+.2f})\n"


        return jsonify({
            "message": f"✅ Lead submitted. Confidence Score: {score}",
            "confidence_score": score,
            "deal_tier": deal_tier,
            "insight": insight
        })
    
    except Exception as e:
        return jsonify({"message": f"❌ Failed to process lead: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
