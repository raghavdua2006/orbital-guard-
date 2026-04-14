import torch
import torch.nn as nn
import numpy as np

class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMModel:
    def __init__(self):
        self.model = LSTMNet()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def create_sequences(self, data, seq_length=10):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    def train(self, data):
        X, y = self.create_sequences(data)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        for epoch in range(3):
            out = self.model(X)
            loss = self.criterion(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1} Loss: {loss.item()}")

    def predict_future(self, history, steps=5):
        seq = history.copy()
        future = []

        for _ in range(steps):
            inp = torch.tensor(seq[-10:], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                next_val = self.model(inp).numpy()[0]

            future.append(next_val.tolist())
            seq.append(next_val)

        return future