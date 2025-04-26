from flask import Flask, Response
from prometheus_client import start_http_server, Gauge
import prometheus_client
from detect_anomalies import load_trained_model, detect_anomalies
import numpy as np
from config import VICTORIA_METRICS_URL, START, END
from data_fetcher import get_victoriametrics_data
from prometheus_client import CollectorRegistry, Gauge, generate_latest

# ساخت متغیرهای Prometheus
cpu_anomaly_gauge = Gauge('cpu_anomaly_detected', 'CPU Anomaly Detected (1: yes, 0: no)')
ram_anomaly_gauge = Gauge('ram_anomaly_detected', 'RAM Anomaly Detected (1: yes, 0: no)')

# Flask app initialization
app = Flask(__name__)

# بارگذاری مدل LSTM آموزش‌دیده
model = load_trained_model('/home/arash/Desktop/entezami/LSTM/lstm_model.h5')

def fetch_data_and_detect_anomalies():
    """
    داده‌ها رو از Victoriametrics می‌گیره و ناهنجاری‌ها رو تشخیص می‌ده.
    """
    # دریافت داده‌ها از Victoriametrics
    raw_data = get_victoriametrics_data(START, END)

    if not raw_data:
        print("❌ No data fetched.")
        return

    # فرض می‌کنیم که داده‌های CPU و RAM به صورت جداگانه ارسال می‌شن
    cpu_data = [float(value[1]) for value in raw_data['data']['result'][0]['values']]
    ram_data = [float(value[1]) for value in raw_data['data']['result'][1]['values']]

    # تشخیص ناهنجاری‌ها
    cpu_anomalies = detect_anomalies(model, cpu_data)
    ram_anomalies = detect_anomalies(model, ram_data)

    # تنظیم مقادیر Prometheus
    cpu_anomaly = 1 if any([anomaly == 1 for _, anomaly in cpu_anomalies]) else 0
    ram_anomaly = 1 if any([anomaly == 1 for _, anomaly in ram_anomalies]) else 0

    # ارسال مقادیر به Prometheus
    cpu_anomaly_gauge.set(cpu_anomaly)
    ram_anomaly_gauge.set(ram_anomaly)


registry = CollectorRegistry()

cpu_anomaly_gauge = Gauge('cpu_anomaly_detected', 'CPU Anomaly Detected (1: yes, 0: no)', registry=registry)
ram_anomaly_gauge = Gauge('ram_anomaly_detected', 'RAM Anomaly Detected (1: yes, 0: no)', registry=registry)


@app.route('/metrics')
def metrics():
    """
    این API برای نمایش داده‌ها و مقادیر Prometheus در دسترس است.
    """
    fetch_data_and_detect_anomalies()
    return Response(prometheus_client.generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    # شروع HTTP server برای Prometheus
    start_http_server(8000)  # این سرور روی پورت 8000 قرار می‌گیره
    app.run(host='0.0.0.0', port=5000)
