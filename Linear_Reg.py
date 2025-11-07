import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import timedelta

# 1️⃣ Load and prepare data
data = pd.read_csv("data/case_time_series.csv", parse_dates=["Date_YMD"])
data = data[["Date_YMD", "Daily Confirmed"]].dropna()

# Convert dates to integer days since start (for regression)
data["Day_Count"] = (data["Date_YMD"] - data["Date_YMD"].min()).dt.days

X = data[["Day_Count"]]
y = data["Daily Confirmed"]

# 2️⃣ Train/Test split
train_size = int(len(data) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 3️⃣ Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4️⃣ Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 5️⃣ Future prediction (next 30 days)
future_days = 30
last_day = int(X["Day_Count"].max())
future_X = np.arange(last_day + 1, last_day + future_days + 1).reshape(-1, 1)
future_dates = [data["Date_YMD"].max() + timedelta(days=i) for i in range(1, future_days + 1)]
future_preds = model.predict(future_X)

# 6️⃣ Visualization
plt.figure(figsize=(10, 5))
plt.scatter(data["Date_YMD"], y, color="lightgray", label="Actual Data")
plt.plot(data["Date_YMD"], model.predict(X), color="blue", linewidth=2, label="Regression Line")
plt.plot(future_dates, future_preds, color="red", linestyle="--", marker="o", label="Future Forecast (30 days)")

plt.title("Linear Regression — COVID-19 Daily Confirmed Cases Forecast")
plt.xlabel("Date")
plt.ylabel("Daily Confirmed Cases")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
