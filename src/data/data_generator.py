"""
Synthetic retail sales data generator — 2 years, 1200 SKUs, 12 stores.
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

STORES = [f"STORE_{i:02d}" for i in range(1, 13)]
SKUS = [f"SKU_{i:04d}" for i in range(1, 1201)]
CATEGORIES = ["Electronics", "Apparel", "Food", "Home", "Sports"]
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")

rows = []
for sku in np.random.choice(SKUS, 50, replace=False):  # sample 50 for speed
    category = np.random.choice(CATEGORIES)
    base_demand = np.random.uniform(30, 200)
    for store in np.random.choice(STORES, 3, replace=False):
        store_factor = np.random.uniform(0.6, 1.4)
        for dt in dates:
            trend = 1 + 0.0003 * (dt - dates[0]).days
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * dt.dayofyear / 365)
            weekly = 1 + 0.2 * np.sin(2 * np.pi * dt.dayofweek / 7)
            is_promotion = int(np.random.random() < 0.08)
            promo_boost = 1.5 if is_promotion else 1.0
            is_holiday = int(dt.month == 12 and dt.day >= 20)
            holiday_boost = 1.4 if is_holiday else 1.0
            noise = np.random.normal(1, 0.05)
            demand = max(0, int(base_demand * store_factor * trend * seasonal * weekly *
                                promo_boost * holiday_boost * noise))
            rows.append({
                "date": dt, "sku_id": sku, "store_id": store, "category": category,
                "units_sold": demand, "is_promotion": is_promotion, "is_holiday": is_holiday,
                "unit_price": round(np.random.uniform(5, 150), 2),
            })

df = pd.DataFrame(rows)
out = Path("data")
out.mkdir(exist_ok=True)
df.to_csv(out / "sales.csv", index=False)
print(f"Generated {len(df):,} rows of sales data")
print(df.head())
