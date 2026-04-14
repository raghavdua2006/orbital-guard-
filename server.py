from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np

from core.data_loader import load_data
from core.lstm_model import LSTMModel
from core.ai_model import AIModel
from core.decision_engine import suggest_action
from core.failure_graph import cascade

# ✅ Create app
app = Flask(__name__)
CORS(app)

# ✅ Home route
@app.route("/")
def home():
    return "🚀 Orbital Guard AI Running"

# ✅ Load data
print("Loading data...")
data, scaler = load_data()

# 🔥 FIX NaN VALUES (VERY IMPORTANT)
data = np.nan_to_num(data, nan=0.5)

# ✅ Train models
print("Training LSTM...")
lstm = LSTMModel()
lstm.train(data)

print("Training AI...")
ai = AIModel()
ai.train(data)

# ✅ API route
@app.route("/api/ai")
def get_data():
    print("----- NEW REQUEST -----")

    res = []

    import time
    global_offset = int(time.time()) % (len(data) - 100)
    
    for i in range(6):
        # Add a satellite-specific offset for more variety
        sat_variance = (i * 20) + (i * i * 2) 
        idx = (global_offset + sat_variance) % (len(data) - 20)

        history = data[idx:idx+10]
        current = data[idx+10]

        # 🔥 HARD CLEAN (fix ANY NaN / inf)
        current = np.nan_to_num(current, nan=0.5, posinf=1.0, neginf=0.0)

        # Extract safely
        temp = float(current[0]) * 100
        battery = float(current[1]) * 100
        signal = float(current[2])

        # 🔥 SAFE AI PREDICTION
        try:
            prediction_label = ai.predict(current)
        except:
            prediction_label = "NOMINAL"

        # Convert label → score
        if prediction_label == "CRITICAL":
            pred_score = 0.9
        elif prediction_label == "DEGRADED":
            pred_score = 0.6
        else:
            pred_score = 0.2

        # 🔥 SAFE LSTM
        try:
            future = lstm.predict_future(history.tolist())
        except:
            future = [temp] * 5

        # Unified Rule Logic - Favor AI Intelligence
        if prediction_label == "CRITICAL" or (temp > 95 and battery < 20):
            status = "CRITICAL"
        elif prediction_label == "DEGRADED" or (temp > 85 or battery < 35):
            status = "DEGRADED"
        else:
            status = "NOMINAL"

        # Safe risk
        try:
            risk = cascade(current[0], current[1], current[2])
        except:
            risk = 0.2

        # Safe actions
        try:
            actions = suggest_action(current[0], current[1], current[2])
        except:
            actions = ["Monitor system"]

        print(f"SAT-{i+1} | Temp:{temp} | Battery:{battery} | Signal:{signal}")

        res.append({
            "satellite_id": f"SAT-{i+1}",
            "thermal": temp,
            "power": battery,
            "signal": signal,

            "prediction_label": prediction_label,
            "prediction_score": pred_score,

            "future": future,
            "risk": risk,
            "status": status,
            "actions": actions
        })

    return jsonify(res)
# ✅ Run server
if __name__ == "__main__":
    app.run(port=5001)