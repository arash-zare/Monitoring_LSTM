### 📈 System Flowchart: SARIMA-EE-LSTM with Fuzzy Logic for Real-Time Anomaly Detection

```mermaid
flowchart TD

A[Start: Fetch Real-Time Metrics from VictoriaMetrics] --> B[Preprocessing: Normalize & Create Sequence]
B --> C[SARIMA Prediction: Forecast Short-Term Trend]
C --> D[Compute Residuals: Actual minus SARIMA Prediction]
D --> E[LSTM Prediction on Residuals: Predict Next Error]
E --> F[Compute MSE between Predicted and Actual Residuals]
F --> G[Pass MSE to Fuzzy Logic System]
G --> H[Fuzzy Inference Engine: Evaluate Anomaly Likelihood]
H --> I{Is It an Anomaly?}
I -->|Yes| J[Flag as Anomaly & Update Prometheus Metric]
I -->|No| K[Mark as Normal & Update Prometheus Metric]
J --> L[Forecast Future Values]
K --> L
L --> M[Expose All Metrics via Flask API at /metrics]
M --> N[Prometheus Scrapes Metrics for Grafana Visualization]
N --> O[Repeat Process at Defined Interval]
