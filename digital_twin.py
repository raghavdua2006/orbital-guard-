class DigitalTwin:
    def __init__(self, lstm_model):
        self.lstm = lstm_model

    def simulate(self, history, steps=5):
        future = []
        seq = history.copy()

        for _ in range(steps):
            next_val = self.lstm.predict_next(seq[-10:])
            future.append({
                "temperature": float(next_val[0]),
                "battery": float(next_val[1]),
                "signal": float(next_val[2])
            })
            seq.append(next_val)

        return future