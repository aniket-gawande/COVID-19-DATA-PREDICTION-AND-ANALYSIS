import warnings
warnings.filterwarnings("ignore")

import os
from datetime import timedelta
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ARIMA import
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception as e:
    ARIMA_AVAILABLE = False
    print("statsmodels ARIMA not available. Install statsmodels to run ARIMA.")
    print("Error:", e)

# ---------- helpers ----------
def evaluate(y_true, y_pred):
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = float("nan")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": r2, "mae": mae, "rmse": rmse}

def fmt(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:,.2f}"

# ---------- Config ----------
DATA_PATH = os.path.join("data", "case_time_series.csv")
FUTURE_DAYS = 30
ARIMA_ORDER = (5, 1, 2)  # baseline; change if you want

# ---------- Load data ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Expected dataset at {DATA_PATH}. Please place case_time_series.csv in ./data/")

df = pd.read_csv(DATA_PATH, parse_dates=["Date_YMD"])
if "Date_YMD" not in df.columns or "Daily Confirmed" not in df.columns:
    raise ValueError("CSV must contain 'Date_YMD' and 'Daily Confirmed' columns")

df = df[["Date_YMD", "Daily Confirmed"]].dropna().sort_values("Date_YMD").reset_index(drop=True)
df["Day_Count"] = (df["Date_YMD"] - df["Date_YMD"].min()).dt.days

# features/target and split
y = df["Daily Confirmed"].values
dates = df["Date_YMD"]
train_frac = 0.8
train_size = int(len(df) * train_frac)

y_train = y[:train_size]
y_test = y[train_size:]
dates_test = dates.iloc[train_size:].reset_index(drop=True)

last_day_count = int(df["Day_Count"].max())
future_dates = [df["Date_YMD"].max() + timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]

results = {}

if not ARIMA_AVAILABLE:
    print("Skipping ARIMA (statsmodels not installed).")
    results["ARIMA"] = {"r2": None, "mae": None, "rmse": None}
else:
    try:
        print(f"\nFitting ARIMA{ARIMA_ORDER} on train series (n={len(y_train)}) ...")
        model = ARIMA(y_train, order=ARIMA_ORDER)
        fit = model.fit()
        print("ARIMA fitted. Forecasting test period...")

        # forecast for test period
        pred_test = fit.forecast(steps=len(y_test))
        metrics = evaluate(y_test, pred_test)
        results["ARIMA"] = metrics

        # save test predictions
        df_test_pred = pd.DataFrame({"Date": dates_test, "Predicted": np.asarray(pred_test).astype(float)})
        df_test_pred.to_csv("arima_test_predictions.csv", index=False)
        print("Saved arima_test_predictions.csv")

        # Refit ARIMA on full series (train+test) for future forecast
        print("Refitting ARIMA on full series to forecast future days...")
        model_full = ARIMA(y, order=ARIMA_ORDER)
        fit_full = model_full.fit()
        pred_future = fit_full.forecast(steps=FUTURE_DAYS)
        df_future = pd.DataFrame({"Date": future_dates, "Predicted": np.asarray(pred_future).astype(float)})
        df_future.to_csv("arima_future_forecast.csv", index=False)
        print("Saved arima_future_forecast.csv")

        print("\nARIMA evaluation on test set:")
        print(f"R2:   {fmt(metrics['r2'])}")
        print(f"MAE:  {fmt(metrics['mae'])}")
        print(f"RMSE: {fmt(metrics['rmse'])}")

    except Exception as e:
        print("ARIMA training/forecast error:", e)
        results["ARIMA"] = {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}

# Print a small table-like summary
print("\n| Model | R² Score | MAE | RMSE |")
print("| ----- | -------- | --- | ---- |")
v = results.get("ARIMA", {"r2": None, "mae": None, "rmse": None})
print(f"| ARIMA | {fmt(v['r2'])} | {fmt(v['mae'])} | {fmt(v['rmse'])} |")

print("\nDone.")
