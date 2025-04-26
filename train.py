import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from model import build_lstm_model
from tensorflow.keras.models import save_model

def prepare_data(data, timesteps=30):
    """
    آماده سازی داده برای مدل LSTM:
    - نرمال سازی
    - ساخت دنباله (Sequence) ورودی و خروجی
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(timesteps, len(scaled_data)):
        x.append(scaled_data[i-timesteps:i])
        y.append(scaled_data[i])

    return np.array(x), np.array(y), scaler

def train_lstm_model(csv_file_path):
    """
    آموزش مدل LSTM با داده‌های ورودی.
    """
    # خواندن داده‌ها
    df = pd.read_csv(csv_file_path)

    # فقط ستون مورد نیاز
    data = df[['cpu_usage']].values

    # آماده‌سازی
    x_train, y_train, scaler = prepare_data(data)

    # ساخت مدل
    model = build_lstm_model((x_train.shape[1], x_train.shape[2]))

    # آموزش مدل
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # ذخیره مدل
    model.save('lstm_model.h5')

    # ذخیره Scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler

if __name__ == "__main__":
    train_lstm_model('cpu_data.csv')


#     آماده‌سازی داده‌ها:

#         داده‌ها به نرمال می‌شن با MinMaxScaler تا برای مدل LSTM مناسب بشن.

#         داده‌ها به صورت دنباله‌ای از timesteps برای یادگیری LSTM آماده می‌شن.

#     آموزش مدل LSTM:

#         مدل با build_lstm_model ساخته می‌شه و سپس با داده‌ها آموزش داده می‌شه.

#     ذخیره مدل:

#         مدل آموزش‌دیده ذخیره می‌شه تا در آینده استفاده بشه.

