import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path="data/nasa.csv"):
    df = pd.read_csv(path)

    # Only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Pad with random if we don't have 3 columns
    for i in range(df.shape[1], 3):
        df[f'dummy_{i}'] = np.random.rand(len(df))

    # Take first 3 columns
    df = df.iloc[:, :3]

    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)

    return data, scaler