# prophet_model_with_visualization.py
"""
Prophet forecasting with Matplotlib visualization (standalone script)

- Requires: prophet, pandas, numpy, scikit-learn, matplotlib
- Input: data/case_time_series.csv with ['Date_YMD', 'Daily Confirmed']
- Output:
    - A plot showing actual data, model fit, and future forecast.
    - Console output with model performance metrics.
    - CSV files: prophet_test_predictions.csv, prophet_future_forecast.csv
- Run: python prophet_model_with_visualization.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Prophet import with error handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install 'prophet' to run this script.")
    print("Error:", e)

# ---------- Helper Functions ----------
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

# ---------- Configuration ----------
DATA_PATH = os.path.join("data", "case_time_series.csv")
FUTURE_DAYS = 30

# ---------- Main Execution ----------
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}. Please place case_time_series.csv in ./data/")

    # 1. Load and prepare data
    df = pd.read_csv(DATA_PATH, parse_dates=["Date_YMD"])
    if "Date_YMD" not in df.columns or "Daily Confirmed" not in df.columns:
        raise ValueError("CSV must contain 'Date_YMD' and 'Daily Confirmed' columns")
    
    df = df[["Date_YMD", "Daily Confirmed"]].dropna().sort_values("Date_YMD").reset_index(drop=True)
    df = df.rename(columns={"Date_YMD": "ds", "Daily Confirmed": "y"})

    # 2. Train/Test split
    train_frac = 0.8
    train_size = int(len(df) * train_frac)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    results = {}

    if not PROPHET_AVAILABLE:
        print("Skipping Prophet execution (library not installed).")
    else:
        try:
            # 3. Train model on the training set for evaluation
            print(f"\nFitting Prophet on train series (n={len(train_df)}) ...")
            m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
            try:
                m.add_country_holidays(country_name='IN')
            except Exception:
                pass # Ignore if holiday data isn't available
            
            m.fit(train_df)
            print("Prophet fitted. Predicting for test period...")

            # 4. Evaluate on the test set
            future_test_df = m.make_future_dataframe(periods=len(test_df), freq='D')
            forecast_test = m.predict(future_test_df)
            pred_test = forecast_test['yhat'].iloc[-len(test_df):].values

            metrics = evaluate(test_df['y'].values, pred_test)
            results["Prophet"] = metrics

            df_test_pred = pd.DataFrame({"Date": test_df['ds'], "Predicted": pred_test.astype(float)})
            df_test_pred.to_csv("prophet_test_predictions.csv", index=False)
            print("Saved prophet_test_predictions.csv")

            print("\nProphet evaluation on test set:")
            print(f"R² Score: {fmt(metrics['r2'])}")
            print(f"MAE:      {fmt(metrics['mae'])}")
            print(f"RMSE:     {fmt(metrics['rmse'])}")

            # 5. Refit on full data to generate final forecast
            print("\nRefitting Prophet on full data to forecast future days...")
            m_full = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
            try:
                m_full.add_country_holidays(country_name='IN')
            except Exception:
                pass
            m_full.fit(df)
            
            future_full_df = m_full.make_future_dataframe(periods=FUTURE_DAYS, freq='D')
            forecast_full = m_full.predict(future_full_df)

            future_forecast_output = forecast_full.iloc[-FUTURE_DAYS:][['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted'})
            future_forecast_output.to_csv("prophet_future_forecast.csv", index=False)
            print("Saved prophet_future_forecast.csv")

            # 6. Visualization
            print("\nGenerating plot...")
            plt.figure(figsize=(12, 6))

            # Plot actual data
            plt.scatter(df['ds'], df['y'], color="gray", alpha=0.5, label="Actual Data")
            
            # Plot the overall model fit (historical + future)
            plt.plot(forecast_full['ds'], forecast_full['yhat'], color="blue", linewidth=2, label="Prophet Model Fit")
            
            # Highlight the future forecast part
            future_plot_df = forecast_full.iloc[len(df):]
            plt.plot(future_plot_df['ds'], future_plot_df['yhat'], color="red", linestyle="--", marker="o", markersize=4, label=f"Future Forecast ({FUTURE_DAYS} days)")
            
            # Plot the confidence interval
            plt.fill_between(forecast_full['ds'], forecast_full['yhat_lower'], forecast_full['yhat_upper'], color='blue', alpha=0.15, label="Confidence Interval (95%)")
            
            plt.title("Prophet Forecast — COVID-19 Daily Confirmed Cases")
            plt.xlabel("Date")
            plt.ylabel("Daily Confirmed Cases")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("An error occurred during Prophet execution:", e)

    print("\nDone.")
