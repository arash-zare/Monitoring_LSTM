# data_fetcher.py
import requests
from config import VICTORIA_METRICS_URL, PROMQL_QUERIES
from datetime import datetime

def get_victoriametrics_data(start: datetime, end: datetime):
    """
    دریافت داده‌ها از ویکتوریا متریکس برای متریک‌های مشخص‌شده در PROMQL_QUERIES.
    """
    result = {
        "data": {
            "result": []
        }
    }

    for metric_name, query in PROMQL_QUERIES.items():
        response = requests.get(
            f"{VICTORIA_METRICS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": int(start.timestamp()),
                "end": int(end.timestamp()),
                "step": "30"
            }
        )
        if response.status_code == 200:
            json_data = response.json()
            if "data" in json_data and "result" in json_data["data"]:
                result["data"]["result"].append(json_data["data"]["result"][0])
        else:
            print(f"❌ خطا در دریافت داده برای {metric_name}: {response.status_code}")

    return result
