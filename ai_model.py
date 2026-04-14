from sklearn.ensemble import IsolationForest

class AIModel:
    def __init__(self):
        # contamination=0.1 for more sensitive detection in high-precision space environments
        self.model = IsolationForest(contamination=0.1, random_state=42)

    def train(self, data):
        self.model.fit(data)

    def predict(self, x):
        # Return status based on anomaly score
        score = self.model.decision_function([x])[0]
        if score < -0.1:
            return "CRITICAL"
        elif score < 0.05:
            return "DEGRADED"
        else:
            return "NOMINAL"