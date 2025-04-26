import numpy as np
from tensorflow.keras.models import load_model

def load_trained_model(model_path='/home/arash/Desktop/entezami/LSTM/lstm_model.h5'):
    """
    مدل آموزش‌دیده LSTM رو بارگذاری می‌کنه.
    """
    model = load_model(model_path)
    return model


def detect_anomaly(model, data, timesteps=30, threshold=0.01):
    """
    تشخیص ناهنجاری با استفاده از مدل LSTM:
    - پیش‌بینی مقادیر بعدی داده‌ها
    - اگر پیش‌بینی از حد آستانه عبور کنه، ناهنجاری تشخیص داده می‌شه.
    """
    # آماده‌سازی داده‌ها
    scaled_data = data[-timesteps:]  # آخرین timesteps داده رو می‌گیریم
    scaled_data = scaled_data.reshape((1, timesteps, 1))

    # پیش‌بینی مقدار بعدی
    prediction = model.predict(scaled_data)

    # بررسی اینکه آیا پیش‌بینی از حد آستانه عبور کرده
    if abs(prediction - data[-1]) > threshold:
        return 1  # Anomaly detected
    else:
        return 0  # No anomaly

def detect_anomalies(model, data, timesteps=30, threshold=0.01):
    """
    تشخیص ناهنجاری برای مجموعه‌ای از داده‌ها.
    """
    anomalies = []
    for i in range(timesteps, len(data)):
        anomaly = detect_anomaly(model, data[:i], timesteps, threshold)
        anomalies.append((data[i], anomaly))  # داده و وضعیت ناهنجاری (0 یا 1)
    return anomalies
