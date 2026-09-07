"""
02_carbon_intensity_forecasting.py

Trains an LSTM to forecast Bangladesh grid carbon intensity 1-6 hours
ahead, using a 48-hour lookback window. Compares against two baselines:
  - Persistence (naive): forecast = last observed value
  - Linear Regression on lagged features

Outputs:
  - ../results/forecast_metrics.csv          (MAE/RMSE/MAPE per model, per horizon)
  - ../results/forecast_vs_actual.png        (plot, test period)
  - ../results/test_period_forecasts.csv     (used later by the scheduler simulation)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "/home/claude/gcp_real/data/bd_grid_carbon_intensity_real.csv"
RESULTS_DIR = "/home/claude/gcp_real/results"
LOOKBACK = 48       # hours of history used as input
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
    """Build (X, y) sliding windows. prior_tail: last LOOKBACK rows of the
    previous split, so val/test windows can see history across the split
    boundary."""
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
    idx = np.array(idx) - offset  # index into the ORIGINAL (non-prefixed) frame
    return X, y, idx

X_train, y_train, _ = make_windows(train_df)
X_val, y_val, _ = make_windows(val_df, prior_tail=train_df.tail(LOOKBACK))
X_test, y_test, test_idx = make_windows(test_df, prior_tail=val_df.tail(LOOKBACK))

print(f"Train windows: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
# 2. LSTM model
# ---------------------------------------------------------------
class CarbonLSTM(nn.Module):
    def __init__(self, n_features, hidden=32, horizon=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1,
                             batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CarbonLSTM(n_features=len(feature_cols)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.MSELoss()

# Normalize the regression target (raw values are ~600-680, far from the
# near-zero default output of an untrained linear head -> without this the
# network converges very slowly).
y_mean, y_std = y_train.mean(), y_train.std()
y_train_n = (y_train - y_mean) / y_std
y_val_n = (y_val - y_mean) / y_std

Xtr = torch.tensor(X_train).to(device)
ytr = torch.tensor(y_train_n).to(device)
Xva = torch.tensor(X_val).to(device)
yva = torch.tensor(y_val_n).to(device)

import time
batch_size = 1024
n_epochs = 20
n_batches = int(np.ceil(len(Xtr) / batch_size))

best_val = np.inf
best_state = None
for epoch in range(1, n_epochs + 1):
    t0 = time.time()
    model.train()
    perm = torch.randperm(len(Xtr))
    total_loss = 0.0
    for b in range(n_batches):
        b_idx = perm[b * batch_size:(b + 1) * batch_size]
        xb, yb = Xtr[b_idx], ytr[b_idx]
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(xb)
    train_loss = total_loss / len(Xtr)

    model.eval()
    with torch.no_grad():
        val_pred = model(Xva)
        val_loss = loss_fn(val_pred, yva).item()
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"Epoch {epoch:2d}  train_MSE={train_loss:8.2f}  val_MSE={val_loss:8.2f}  ({time.time()-t0:.1f}s)", flush=True)

model.load_state_dict(best_state)

# ---------------------------------------------------------------
# 3. Evaluate LSTM on test set
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    Xte = torch.tensor(X_test).to(device)
    lstm_pred_n = model(Xte).cpu().numpy()
lstm_pred = lstm_pred_n * y_std + y_mean  # back to gCO2/kWh

# ---------------------------------------------------------------
# 4. Baselines
# ---------------------------------------------------------------
# Persistence: forecast every horizon step = last known value (X_test[:, -1, 0]
# is the scaled last carbon_intensity in the window; unscale it)
last_scaled = X_test[:, -1, 0]
ci_mean, ci_std = scaler.mean_[0], scaler.scale_[0]
last_actual = last_scaled * ci_std + ci_mean
persistence_pred = np.repeat(last_actual[:, None], HORIZON, axis=1)

# Linear Regression on flattened lookback window -> horizon vector
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)
linreg = LinearRegression()
linreg.fit(X_train_flat, y_train)
linreg_pred = linreg.predict(X_test_flat)

# ---------------------------------------------------------------
# 5. Metrics per horizon step
# ---------------------------------------------------------------
def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

rows = []
for h in range(HORIZON):
    for name, pred in [("LSTM", lstm_pred), ("LinearRegression", linreg_pred),
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
# 6. Save test-period forecasts (1-hour-ahead) for the scheduler sim
# ---------------------------------------------------------------
test_timestamps = test_df["timestamp"].values[test_idx]
forecast_export = pd.DataFrame({
    "timestamp": test_timestamps,
    "actual_ci": y_test[:, 0],
})
for h in range(HORIZON):
    forecast_export[f"lstm_forecast_h{h+1}"] = lstm_pred[:, h]
forecast_export.to_csv(f"{RESULTS_DIR}/test_period_forecasts.csv", index=False)

# ---------------------------------------------------------------
# 7. Plot: 1-hour-ahead forecast vs actual, last 7 days of test set
# ---------------------------------------------------------------
plot_n = 24 * 7
plt.figure(figsize=(11, 4.5))
plt.plot(test_timestamps[:plot_n], y_test[:plot_n, 0], label="Actual", linewidth=1.6)
plt.plot(test_timestamps[:plot_n], lstm_pred[:plot_n, 0], label="LSTM (1h-ahead)", linewidth=1.3)
plt.plot(test_timestamps[:plot_n], persistence_pred[:plot_n, 0], label="Persistence", linewidth=1, linestyle="--", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Carbon intensity (gCO2/kWh)")
plt.title("1-Hour-Ahead Carbon Intensity Forecast vs Actual (first 7 days of test period)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/forecast_vs_actual.png", dpi=150)
print(f"\nSaved plot to {RESULTS_DIR}/forecast_vs_actual.png")
