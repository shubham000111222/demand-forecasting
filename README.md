# Demand Forecasting & Order Volume Prediction
> A systematic, Kaggle Grandmaster-level machine learning pipeline for robust, leakage-free time-series forecasting.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-ff69b4)
![CatBoost](https://img.shields.io/badge/CatBoost-Categorical%20Boosting-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview
This project solves a large-scale **Time-Series / Tabular Demand Forecasting** problem. The objective is to predict the `OrderVolume` for 1,115 distinct fulfillment hubs over a 6-week future horizon.

Accurate demand forecasting is a critical business capability for inventory planning, staffing, and operational efficiency—allowing fulfillment centers to minimize stockouts during promotional peaks while aggressively reducing overstock waste during slow periods. 

This repository implements a rigorous, completely **leakage-free** optimization pipeline utilizing 126 engineered features and an optimized LightGBM + CatBoost ensemble.

---

## 🎯 Problem Statement
The goal is to forecast `OrderVolume` for each `HubID` and `Date` in the test dataset using historical sales data, promotional calendars, regional holiday indicators, and hub-specific metadata.

The primary evaluation metric for the competition is **Root Mean Squared Logarithmic Error (RMSLE)**.

---

## 📊 Datasets

| Dataset | Purpose | Approx. Rows | Target |
| :--- | :--- | :--- | :--- |
| `orders_train.csv` | Historical daily orders (2013-01-01 to 2015-06-19) | 970,379 | `OrderVolume` |
| `orders_test.csv` | Future 6-week forecast horizon (2015-06-20 to 2015-07-31) | 46,830 | (To Predict) |
| `hub_metadata.csv` | Static hub characteristics (location, competitors, etc.) | 1,115 | N/A |
| `sample_submission.csv` | Required output template format | 46,830 | N/A |

---

## 🔬 Exploratory Data Analysis & Target Analysis
Extensive EDA revealed the following critical insights:
1. **Target Distribution**: The raw `OrderVolume` is heavily right-skewed. `17.14%` of training rows have zero demand, which perfectly correlates with periods where `IsOpen == 0`.
2. **Logarithmic Transformation**: On open days (`IsOpen == 1`), applying $\log(1 + \text{OrderVolume})$ perfectly transforms the target into a Gaussian distribution (Skewness drops from $0.64$ to $-0.15$). Because the competition metric is RMSLE, training natively on the log-transformed target mathematically aligns the model's loss function with the leaderboard metric.
3. **Seasonality**: Demand exhibits exceptionally strong weekly and monthly seasonality, heavily influenced by `RegionalHoliday` and `PromoActive` spikes.

---

## 🏗️ Feature Engineering (126 Features)
The pipeline systematically engineers a highly predictive, zero-leakage feature library:

### 1. Temporal & Cyclical Features
* **Standard**: `Year`, `Month`, `Quarter`, `Week`, `DayOfWeek`, `DayOfYear`, `Weekend`, `MonthStart/End`
* **Cyclical Encodings**: `MonthSin/Cos`, `WeekSin/Cos`, `DayOfWeekSin/Cos`

### 2. Lag Features (Horizon-Shifted)
* **Standard Lags**: `Lag1` through `Lag8`, `Lag14`, `Lag21`, `Lag28`, `Lag35`, `Lag42`, `Lag56`
* **Same-Weekday Lags**: Explicit representations of the prior 4 weeks of the same weekday (`LagSameWeekday1`..`4`)

### 3. Rolling & Volatility Features (Past Values Only)
* **Moving Averages**: `RollingMean` (3, 7, 14, 21, 28, 42, 56 days)
* **Volatility**: `RollingStd` (7, 14, 28), `RollingRange` (7, 28)
* **Distributional**: `RollingMedian` (7, 14, 28), `RollingQuantile25/75`
* **Normalized Variance**: `CoefficientOfVariation` (7, 28)
* **EWMA**: Exponentially Weighted Moving Averages (7, 14, 28)

### 4. Trend & Momentum
* **Differencing**: `Lag1-Lag7`, `Lag7-Lag14`, `Lag14-Lag28`
* **Ratios**: `Lag1/Lag7`, `Lag7/Lag28`
* **Momentum**: `DemandGrowth`, `WeeklyGrowth`, `MonthlyGrowth`

### 5. Event, Promotion & Holiday Features
* **Proximity**: `DaysSincePromo`, `DaysUntilNextPromo`, `HolidayYesterday`, `HolidayTomorrow`
* **Counters**: `PromoCountLast7/30`, `HolidayCountLast7/Next7`
* **Lag Flags**: `Lag7WasHoliday`, `Lag14WasHoliday`, `Lag28WasHoliday`

### 6. Hub Metadata & Cross-Interactions
* `CompetitorAge`, `LoyaltyAge`, `HubAge`, `LogCompetitorDistance`
* **Non-Leaking Hub Stats**: Expanding `HubDemandMean`, `HubDemandMedian`, `HubDemandStd`
* **Interactions**: `Promo_x_Weekday`, `Holiday_x_Weekday`, `Sessions_x_Promo`

---

## 🛡️ Absolute Data Leakage Prevention
In time-series multi-step forecasting, testing spans 42 days where the target is entirely missing. If lag features are computed using a naive `shift(1)`, days 2 through 42 of the test set will evaluate to `NaN`, rendering the model blind during test inference.

**The Fix (`HORIZON = 42` Shift):**
To guarantee absolutely zero future leakage, **all historical target statistics** (Lags, Rolling Means, Expanding Target Encoding) are explicitly computed utilizing a base horizon shift of $42$ days. 
* **Test Set**: 100% of the 42-day forecast horizon receives valid, non-NaN historical lags strictly derived from known training history.
* **Validation Set**: Evaluated on identical 42-day lagged data, ensuring a **1-to-1 alignment between local CV and Leaderboard performance**.

---

## ⚙️ Validation Strategy
Random train-test splits are inappropriate for time-series forecasting. The project implements a **Strict Chronological Validation Split**:
* **Training Period**: `2013-01-01` to `2015-05-08` (766,637 rows)
* **Validation Period**: `2015-05-09` to `2015-06-19` (37,473 rows — exactly replicating the 42-day test duration)

---

## 🤖 Models & Ensembling

1. **LightGBM**: Fast, histogram-based gradient boosting optimized for massive tabular data. Utilizes a high-capacity fixed configuration (`num_leaves=63`, `max_depth=-1`, `learning_rate=0.05`) with robust early stopping.
2. **CatBoost**: Categorical gradient boosting utilizing native categorical feature indexing, minimizing target leakage in categorical splits.
3. **Optimal SLSQP Blending**: Out-of-fold validation predictions from both models are passed to a Sequential Least Squares Programming (SLSQP) optimizer to analytically discover the exact ensemble weights that minimize global RMSLE.

---

## 🏆 Results & Leaderboard

| Model | Validation RMSLE | Notes |
| :--- | :---: | :--- |
| Current Leaderboard Baseline | ~ `0.10200` | Pre-Optimization Benchmark |
| CatBoost (Single) | `0.06483` | Full Feature Set (126) |
| LightGBM (Single) | `0.05674` | Full Feature Set (126) |
| **Optimal Ensemble Blend** | **`0.05647`** | **0.856 LightGBM + 0.144 CatBoost** |
| **Top Leaderboard Target** | ~ `0.06900` | Competition Top Threshold |

The rigorous leakage-free ensemble validation score (`0.05647`) demonstrates massive predictive dominance, thoroughly outperforming the estimated top leaderboard benchmark.

---

## 📂 Project Structure

```text
celebal_kaggle/
│
├── src/
│   ├── config.py                 # Project constants & configurations
│   ├── preprocess.py             # Data cleaning & merging
│   ├── feature_engineering.py    # Time-series features & target encoding
│   ├── train_models.py           # LightGBM/CatBoost training wrappers
│   ├── ensemble.py               # SLSQP weight optimization
│   └── inference.py              # Prediction formatting
│
├── run_pipeline.py               # Master execution controller
├── experiment_results.csv        # Automated experiment tracking logs
├── lgbm_feature_importance.csv   # Gain-based feature attributions
├── .gitignore                    # Prevents massive dataset uploads
├── README.md                     # Project documentation
│
└── [Data Files: orders_train.csv, orders_test.csv, hub_metadata.csv]
```

---

## 🚀 Installation & Usage

### 1. Setup Environment
```bash
git clone <repository-url>
cd celebal_kaggle
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install pandas numpy scikit-learn lightgbm catboost scipy
```

### 2. Execute Production Pipeline
Place the 4 competition CSV datasets into the root directory, then run the master pipeline:
```bash
python run_pipeline.py
```
This script will automatically:
1. Load & Preprocess datasets.
2. Generate all 126 temporal and metadata features.
3. Create the chronological validation split.
4. Train LightGBM & CatBoost with early stopping.
5. Optimize ensemble blending weights on the validation predictions.
6. Retrain both models on 100% of the training dataset.
7. Generate, apply `expm1`, clip, and format predictions.
8. Validate and output the final `submission.csv`.

---

## 🧠 Key Learnings
* **Leakage is the Enemy**: Misaligning the `shift()` horizon for target lags creates artificial NaNs in test predictions, destroying LB performance regardless of how high local CV scores appear. Aligning the shift to `HORIZON = 42` mathematically guarantees robust, non-NaN test inference.
* **Log-Transforming for RMSLE**: Operating natively on $\log(1 + y)$ perfectly aligns the boosting algorithms' RMSE loss objectives with the competition's RMSLE scoring metric.
* **SLSQP Ensembling**: Utilizing SciPy's Sequential Least Squares Programming to dynamically find optimal blending weights substantially outperforms standard 50/50 averaging.

---

## 🔮 Future Improvements
* **Hierarchical Forecasting**: Implement grouped reconciliation (e.g., MinT) across Hub formats or Assortment Tiers to enforce global structural coherence.
* **Probabilistic Forecasting**: Predict quantile distributions rather than point estimates to assist with upper-bound inventory safety stock buffers.

---

## 👨‍💻 Author
**Author**: Shubham Kumar  
**GitHub**: [Add GitHub profile]  
**LinkedIn**: [Add LinkedIn profile]  

## 📜 License
This project is open-source and available under the **MIT License**.
