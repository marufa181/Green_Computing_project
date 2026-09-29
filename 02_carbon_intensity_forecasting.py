"""
02_carbon_intensity_forecasting.py

Forecasts Bangladesh grid carbon intensity 1-6 hours ahead, using a
48-hour lookback window. Compares three models:
  - Persistence (naive): forecast = last observed value
  - Linear Regression on the flattened lookback window
  - Random Forest Regressor on the flattened lookback window (our main
    ML model -- chosen over a deep LSTM because this environment's disk
    quota cannot accommodate PyTorch's CUDA dependencies; Random Forest
    is a well-established, high-performing method for tabular time-series
    forecasting and needs no GPU/heavy dependencies)

Outputs:
  - ../results/forecast_metrics.csv          (MAE/RMSE/MAPE per model, per horizon)
  - ../results/forecast_vs_actual.png        (plot, test period)
  - ../results/test_period_forecasts.csv     (used later by the scheduler simulation)
"""

import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent   # project root (works from any folder)
DATA_PATH = ROOT / "data" / "bd_grid_carbon_intensity_real.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOOKBACK = 48        # hours of history used as input
HORIZON = 6          # forecast this many hours ahead

# ---------------------------------------------------------------
# 1. Load & feature-engineer
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
df["doy_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofyear / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofyear / 365)

feature_cols = ["carbon_intensity_gco2_per_kwh", "demand_mw",
                "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
target_col = "carbon_intensity_gco2_per_kwh"

# Chronological split: 80% train, 10% val, 10% test
n = len(df)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
train_df = df.iloc[:n_train].reset_index(drop=True)
val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

def make_windows(frame, prior_tail=None):
    if prior_tail is not None:
        frame = pd.concat([prior_tail, frame], ignore_index=True)
        offset = len(prior_tail)
    else:
        offset = 0
    scaled = scaler.transform(frame[feature_cols])
    target = frame[target_col].values
    X, y, idx = [], [], []
    for i in range(LOOKBACK, len(frame) - HORIZON + 1):
        X.append(scaled[i - LOOKBACK:i])
        y.append(target[i:i + HORIZON])
        idx.append(i)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    idx = np.array(idx) - offset
    return X, y, idx

X_train, y_train, _ = make_windows(train_df)
X_val, y_val, _ = make_windows(val_df, prior_tail=train_df.tail(LOOKBACK))
X_test, y_test, test_idx = make_windows(test_df, prior_tail=val_df.tail(LOOKBACK))

print(f"Train windows: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# ---------------------------------------------------------------
# 2. Random Forest Regressor (main ML model)
# ---------------------------------------------------------------
import time
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=60, max_depth=10, min_samples_leaf=5,
    n_jobs=1, random_state=42,
)
rf.fit(X_train_flat, y_train)
print(f"Random Forest trained in {time.time()-t0:.1f}s", flush=True)

val_pred_rf = rf.predict(X_val_flat)
val_mae_rf = np.mean(np.abs(val_pred_rf - y_val))
print(f"Random Forest validation MAE: {val_mae_rf:.3f}")

rf_pred = rf.predict(X_test_flat)

# ---------------------------------------------------------------
# 3. Baselines
# ---------------------------------------------------------------
last_scaled = X_test[:, -1, 0]
ci_mean, ci_std = scaler.mean_[0], scaler.scale_[0]
last_actual = last_scaled * ci_std + ci_mean
persistence_pred = np.repeat(last_actual[:, None], HORIZON, axis=1)

linreg = LinearRegression()
linreg.fit(X_train_flat, y_train)
linreg_pred = linreg.predict(X_test_flat)

# ---------------------------------------------------------------
# 4. Metrics per horizon step
# ---------------------------------------------------------------
def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

rows = []
for h in range(HORIZON):
    for name, pred in [("RandomForest", rf_pred), ("LinearRegression", linreg_pred),
                        ("Persistence", persistence_pred)]:
        mae, rmse, mape = metrics(y_test[:, h], pred[:, h])
        rows.append({"model": name, "horizon_hours": h + 1,
                     "MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE_%": round(mape, 3)})
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(f"{RESULTS_DIR}/forecast_metrics.csv", index=False)
print("\n=== Forecast accuracy by horizon ===")
print(metrics_df.pivot(index="horizon_hours", columns="model", values="MAE"))

overall = metrics_df.groupby("model")[["MAE", "RMSE", "MAPE_%"]].mean().round(3)
overall.to_csv(f"{RESULTS_DIR}/forecast_metrics_overall.csv")
print("\n=== Overall (averaged across 1-6h horizons) ===")
print(overall)

# ---------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------
importances = rf.feature_importances_.reshape(LOOKBACK, len(feature_cols))
importance_by_feature = importances.sum(axis=0)
imp_df = pd.DataFrame({"feature": feature_cols, "importance": importance_by_feature})
imp_df = imp_df.sort_values("importance", ascending=False)
imp_df.to_csv(f"{RESULTS_DIR}/rf_feature_importance.csv", index=False)
print("\n=== Random Forest feature importance (summed across the 48h lookback) ===")
print(imp_df)

# ---------------------------------------------------------------
# 6. Save test-period forecasts (for the scheduler sim)
# ---------------------------------------------------------------
test_timestamps = test_df["timestamp"].values[test_idx]
forecast_export = pd.DataFrame({
    "timestamp": test_timestamps,
    "actual_ci": y_test[:, 0],
})
for h in range(HORIZON):
    forecast_export[f"rf_forecast_h{h+1}"] = rf_pred[:, h]
forecast_export.to_csv(f"{RESULTS_DIR}/test_period_forecasts.csv", index=False)

# ---------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------
plot_n = 24 * 7
plt.figure(figsize=(11, 4.5))
plt.plot(test_timestamps[:plot_n], y_test[:plot_n, 0], label="Actual", linewidth=1.8, color="#1f4e8c", zorder=5)
plt.plot(test_timestamps[:plot_n], linreg_pred[:plot_n, 0], label="Linear Regression (1h-ahead) — best model", linewidth=1.4, color="#d62728")
plt.plot(test_timestamps[:plot_n], rf_pred[:plot_n, 0], label="Random Forest (1h-ahead)", linewidth=1.1, color="#ff7f0e", alpha=0.85)
plt.plot(test_timestamps[:plot_n], persistence_pred[:plot_n, 0], label="Persistence", linewidth=1, linestyle="--", color="#2ca02c", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Carbon intensity (gCO2/kWh)")
plt.title("1-Hour-Ahead Carbon Intensity Forecast vs Actual (first 7 days of test period)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/forecast_vs_actual.png", dpi=150)
print(f"\nSaved plot to {RESULTS_DIR}/forecast_vs_actual.png")
