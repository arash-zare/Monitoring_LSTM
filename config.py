# config.py
from datetime import datetime, timedelta

VICTORIA_METRICS_URL = "http://192.168.1.98:8428"

# PromQL queries to fetch metrics - customize as needed
PROMQL_QUERIES = {
    "cpu": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[1m])) * 100)",
    "memory": "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes"
}

# زمان شروع و پایان برای درخواست‌های PromQL
END = datetime.utcnow()
START = END - timedelta(minutes=5)

# Model settings
WINDOW_SIZE = 30  # number of data points per sample
THRESHOLD = 0.001  # MSE threshold to detect anomaly

# Model path (for saving/loading trained LSTM model)
MODEL_PATH = "models/lstm_anomaly_detector.h5"

# Logging settings
LOGGING_LEVEL = "DEBUG"