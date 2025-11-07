import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import timedelta

# 1️⃣ Load dataset
data = pd.read_csv("data/case_time_series.csv", parse_dates=["Date_YMD"])
data = data[["Date_YMD", "Daily Confirmed"]].dropna()

# 2️⃣ Convert date to numeric (days since first date)
data["Day_Count"] = (data["Date_YMD"] - data["Date_YMD"].min()).dt.days
X = data[["Day_Count"]]
y = data["Daily Confirmed"]

# 3️⃣ Split data (80% train, 20% test)
train_size = int(len(data) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 4️⃣ Train Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Predict on test set
y_pred = model.predict(X_test)

# 6️⃣ Evaluate model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("🔹 Decision Tree Performance:")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 7️⃣ Future Forecast (next 30 days)
future_days = 30
last_day = int(X["Day_Count"].max())
future_X = np.arange(last_day + 1, last_day + future_days + 1).reshape(-1, 1)
future_dates = [data["Date_YMD"].max() + timedelta(days=i) for i in range(1, future_days + 1)]
future_preds = model.predict(future_X)

# 8️⃣ Plot Results
plt.figure(figsize=(10, 5))
plt.scatter(data["Date_YMD"], y, color="gray", alpha=0.5, label="Actual Data")
plt.plot(data["Date_YMD"], model.predict(X), color="blue", linewidth=2, label="Decision Tree Fit")
plt.plot(future_dates, future_preds, color="red", linestyle="--", marker="o", label="Future Forecast (30 days)")

plt.title("Decision Tree Regression — COVID-19 Daily Confirmed Cases Forecast")
plt.xlabel("Date")
plt.ylabel("Daily Confirmed Cases")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
