### 📈 System Flowchart: SARIMA-EE-LSTM with Fuzzy Logic for Real-Time Anomaly Detection

```mermaid
flowchart TD

A[Start: Fetch Real-Time Metrics from VictoriaMetrics] --> B[Preprocessing: Normalize & Create Sequence]

B --> C[SARIMA Prediction: Forecast Short-Term Trend]
C --> D[Compute Residuals: Actual - SARIMA_Prediction]

D --> E[LSTM Prediction on Residuals: Predict Next Error]
E --> F[Compute MSE: (Predicted Residual - Actual Residual)^2]

F --> G[Pass MSE to Fuzzy Logic System]
G --> H[Fuzzy Inference: Is this an Anomaly?]
H --> I{Fuzzy Output}

I -->|Yes| J[Flag as Anomaly & Update Prometheus Metric]
I -->|No| K[Mark as Normal & Update Prometheus Metric]

J --> L[Forecast Future Values]
K --> L

L --> M[Expose All Metrics via Flask API /metrics]
M --> N[Prometheus Scrapes Metrics ➝ Grafana Visualization]

N --> O[Repeat Process Every INTERVAL Seconds]
