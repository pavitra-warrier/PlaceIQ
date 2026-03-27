"""
app.py — Smart Placement Guidance System Backend
Run: python app.py
Then open: http://localhost:5000
"""
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load model & scaler ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model  = joblib.load(os.path.join(BASE_DIR, "placement_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "placement_scaler.pkl"))
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print("   Please run: python train_model.py  — to generate the model files first.")
    model, scaler = None, None

ROLE_LABELS = ["SDE", "Data Analyst", "ML Engineer", "Core Engineering", "Non-Tech"]

ROLE_INFO = {
    "SDE": {
        "icon": "💻",
        "tips": ["Strengthen DSA on LeetCode", "Build 2-3 full-stack projects", "Practice system design"],
        "avg_package": "8–20 LPA"
    },
    "Data Analyst": {
        "icon": "📊",
        "tips": ["Learn SQL, Power BI / Tableau", "Work on EDA projects", "Practice statistics & Excel"],
        "avg_package": "6–15 LPA"
    },
    "ML Engineer": {
        "icon": "🤖",
        "tips": ["Master scikit-learn & PyTorch", "Compete on Kaggle", "Build end-to-end ML pipelines"],
        "avg_package": "10–25 LPA"
    },
    "Core Engineering": {
        "icon": "⚙️",
        "tips": ["Revise core subjects (VLSI, Circuits, Mechanics)", "Apply for PSU exams (GATE)", "Get domain certifications"],
        "avg_package": "5–12 LPA"
    },
    "Non-Tech": {
        "icon": "🎯",
        "tips": ["Sharpen communication & leadership", "Prepare for MBA / consulting", "Build a strong LinkedIn profile"],
        "avg_package": "5–10 LPA"
    }
}

# ── CORS helper ──────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Run python train_model.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    required = [
        "cgpa", "backlogs", "semester", "leetcode_rating", "codechef_rating",
        "contests_won", "hackathons", "internships", "projects_count",
        "communication_score", "leadership_score", "ds_score", "ml_score",
        "web_score", "embedded_score"
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features = np.array([[
            float(data["cgpa"]),
            int(data["backlogs"]),
            int(data["semester"]),
            float(data["leetcode_rating"]),
            float(data["codechef_rating"]),
            int(data["contests_won"]),
            int(data["hackathons"]),
            int(data["internships"]),
            int(data["projects_count"]),
            float(data["communication_score"]),
            float(data["leadership_score"]),
            float(data["ds_score"]),
            float(data["ml_score"]),
            float(data["web_score"]),
            float(data["embedded_score"]),
        ]])

        features_scaled = scaler.transform(features)
        preds  = model.predict(features_scaled)[0]
        probas = model.predict_proba(features_scaled)

        placement_label = "Yes" if int(preds[0]) == 1 else "No"
        # probas[0] is shape (1, n_classes) for placement
        placement_conf  = float(np.max(probas[0][0]))

        role_idx   = int(preds[1])
        role_label = ROLE_LABELS[role_idx]
        # probas[1] is shape (1, n_classes) for role
        role_conf  = float(np.max(probas[1][0]))

        # All role probabilities for radar chart
        role_probs = {
            ROLE_LABELS[i]: round(float(probas[1][0][i]), 3)
            for i in range(len(ROLE_LABELS))
        }

        info = ROLE_INFO[role_label]

        return jsonify({
            "placement":        placement_label,
            "confidence":       round(placement_conf, 2),
            "role":             role_label,
            "role_icon":        info["icon"],
            "best_fit_score":   round(role_conf, 2),
            "role_probabilities": role_probs,
            "tips":             info["tips"],
            "avg_package":      info["avg_package"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("\n🚀 Starting Smart Placement Guidance System...")
    print("   Open your browser at: http://localhost:5000\n")
    app.run(debug=True, port=5000)
