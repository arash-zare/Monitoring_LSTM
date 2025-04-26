import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_series(df, window_size=30):
    """
    ورودی: یک سری زمانی (DataFrame با یک ستون `value`)
    خروجی: X, y برای آموزش مدل LSTM
    """

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_values) - window_size):
        X.append(scaled_values[i:i+window_size])
        y.append(scaled_values[i+window_size])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler
