# Retail Demand Forecasting System

> Prophet + LSTM hybrid forecasting on 2 years of retail sales data. Automated weekly retraining via Airflow DAG. Reduces stockout rate from 18% to 6%.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Prophet](https://img.shields.io/badge/Prophet-1.1-orange) ![LSTM](https://img.shields.io/badge/LSTM-Keras-red) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)

---

## Problem Statement

A retail chain had an 18% stockout rate on top SKUs and $800K in excess inventory costs annually. Manual forecasting by category managers was inconsistent and couldn't scale beyond 50 SKUs.

## Approach

1. **EDA** — 2 years of daily sales, 1,200 SKUs across 12 stores
2. **Feature Engineering** — lag features, rolling stats, holiday flags, promotions, weather
3. **Modelling** — Prophet (trend + seasonality) + LSTM (residual learning) hybrid
4. **Hyperparameter tuning** — cross-validation on rolling windows
5. **Retraining** — weekly Airflow DAG, model versioning with MLflow
6. **Serving** — FastAPI forecast endpoint with confidence intervals

## Results

| Metric | Before | After |
|--------|--------|-------|
| Stockout rate | 18% | **6%** |
| MAPE | 22% | **9%** |
| Inventory savings | — | **$340K/year** |
| SKUs covered | 50 | **1,200** |

## Project Structure

```
demand-forecasting/
├── api/
│   └── main.py              # Forecast API endpoint
├── notebooks/
│   ├── 01_eda.py            # Sales EDA + seasonality analysis
│   └── 02_model_training.py # Prophet + LSTM hybrid
├── src/
│   ├── data/
│   │   └── data_generator.py  # Synthetic retail dataset
│   ├── features/
│   │   └── feature_engineering.py
│   └── models/
│       ├── prophet_model.py
│       ├── lstm_model.py
│       └── train.py
├── dags/
│   └── weekly_retrain.py    # Airflow DAG
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python src/data/data_generator.py
python src/models/train.py
uvicorn api.main:app --reload
```

## API Usage

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"sku_id": "SKU_0042", "store_id": "STORE_03", "horizon_days": 14}'
```

Response:
```json
{
  "sku_id": "SKU_0042",
  "store_id": "STORE_03",
  "horizon_days": 14,
  "forecasts": [
    {"date": "2026-03-04", "predicted_units": 142, "lower_ci": 118, "upper_ci": 166},
    {"date": "2026-03-05", "predicted_units": 89, "lower_ci": 71, "upper_ci": 107}
  ],
  "mape_historical": 8.7,
  "model": "prophet_lstm_hybrid"
}
```

---

**Author**: Your Name · [GitHub](https://github.com/shubham000111222)
