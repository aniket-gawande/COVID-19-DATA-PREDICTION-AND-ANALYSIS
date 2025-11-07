# 🦠 COVID-19 Data Prediction & Analysis (India)

### 📊 An Interactive Data Science Dashboard with Machine Learning Forecasting  
**Project Type:** Academic Mini Project (VSEC – Data Science Laboratory)  
**Institution:** Pimpri-Chinchwad College of Engineering (PCCOE), Pune  
**Academic Year:** 2025-2026  

---

## 🌐 Overview

The **COVID-19 India Prediction & Analysis Dashboard** is a comprehensive **data science project** that integrates **interactive visual analytics** and **machine learning forecasting models**.  
It enables users to explore historical COVID-19 data — including confirmed, recovered, deceased, and vaccination metrics — and predict future trends using **ARIMA** and **Prophet** models.

This project was built to help policymakers, researchers, and the general public make **data-driven decisions** through an easy-to-use dashboard powered by **Python and Streamlit**.

---

## 🎯 Objectives

- Collect and preprocess reliable COVID-19 datasets from trusted sources.  
- Perform **Exploratory Data Analysis (EDA)** to identify pandemic trends and regional hotspots.  
- Develop an **interactive web dashboard** for real-time visualization.  
- Implement and compare **machine learning models** for time-series forecasting:
  - Linear Regression
  - Decision Tree Regressor
  - ARIMA
  - Prophet
- Evaluate model performance using **R² Score**, **MAE**, and **RMSE**.  
- Deploy the dashboard for public use with data export and download options.

---

## 🧠 Methodology

### 🔹 Phase 1: Data Acquisition
- Data sourced from [covid19india.org](https://data.covid19india.org), **MOHFW**, and **CoWIN** APIs.  
- Verified via schema and checksum validation for data integrity.

### 🔹 Phase 2: Data Cleaning & Transformation
- Handled missing values and standardized naming conventions.  
- Derived fields such as **Active Cases** dynamically computed.

### 🔹 Phase 3: Dashboard Logic & Filtering
- Multi-level filtering: *India → State → District → Vaccine*.  
- Optimized Pandas queries ensure responsiveness and low latency.

### 🔹 Phase 4: Visualization
- Line, Bar, and Area charts built using **Plotly Express**.  
- Dark-theme UI with color-coded legends for better readability.

### 🔹 Phase 5: Export & Sharing
- “Download CSV” button for one-click data export.

### 🔹 Phase 6: Branding & UI Design
- Custom dark theme, floating footer credits, and professional layout.

---

## 🧰 Tools & Technologies

| Category | Tools Used |
|-----------|-------------|
| **Programming Language** | Python 3.10 |
| **Libraries** | Pandas, NumPy, Plotly, Streamlit, Matplotlib, Scikit-learn |
| **Time-Series Models** | Statsmodels (ARIMA), Prophet (Facebook) |
| **Development Environment** | VS Code, Jupyter Notebook |
| **Version Control** | Git & GitHub |
| **Data Sources** | covid19india.org, MOHFW, CoWIN |

---

## 🧩 Machine Learning Models

| Model | Type | Description |
|--------|------|-------------|
| **Linear Regression** | Baseline | Captures linear trends, used for benchmarking. |
| **Decision Tree Regressor** | Non-linear | Handles complex patterns, limited extrapolation. |
| **ARIMA** | Statistical | Captures temporal dependencies and autocorrelation. |
| **Prophet** | Advanced | Decomposes trend, seasonality, and holidays with confidence intervals. |

### 📈 Model Performance

| Model | R² Score | MAE | RMSE |
|-------|-----------|------|------|
| Linear Regression | -1.183 | 68,912.45 | 86,110.13 |
| Decision Tree | -0.428 | 55,270.83 | 70,051.49 |
| ARIMA | 0.695 | 18,450.76 | 28,980.21 |
| Prophet | **0.812** | **14,215.33** | **22,950.88** |

> 🧩 **Prophet outperformed all models**, capturing seasonal and trend variations effectively.

---

## 📊 30-Day Forecast (Example)

| Horizon | Prophet Predicted Cases | ARIMA Predicted Cases | Trend |
|----------|------------------------|-----------------------|--------|
| +7 Days  | 22,000 – 24,000 | 25,000 – 27,000 | Gradual Decline |
| +14 Days | 18,000 – 20,000 | 21,000 – 23,000 | Stabilizing |
| +30 Days | 14,000 – 16,000 | 18,000 – 20,000 | Near Baseline |

---

## 🚀 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/aniket-gawande/covid19-data-analysis.git
cd covid19-data-analysis
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
pandas
numpy
plotly
streamlit
matplotlib
scikit-learn
statsmodels
prophet
```

### 3️⃣ Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
COVID19-Data-Prediction/
│
├── data/
│   ├── case_time_series.csv
│   ├── states.csv
│   ├── districts.csv
│   └── vaccine_doses_statewise_v2.csv
│
├── app.py                    # Main Streamlit Dashboard
├── linear_regression.py      # Baseline model
├── decision_tree.py          # Non-linear model
├── arima_model.py            # ARIMA time-series model
├── prophet_model.py          # Prophet forecast model
├── requirements.txt
└── README.md
```

---

## 📘 Key Insights

* Strong positive correlation between **Daily Confirmed** and **Daily Recovered (r ≈ 0.93)**
* Time-series data is **non-random**, validating ARIMA/Prophet use.
* **Wave-2 CFR (Case Fatality Rate)** statistically higher than Wave-1 (p < 0.05).
* Prophet’s confidence intervals enhance **risk awareness** and **policy decisions**.

---

## 🧩 Future Enhancements

* Integrate **LSTM/GRU Deep Learning Models** for multivariate forecasting.
* Add **real-time API integration** for live updates.
* Include **interactive geo-maps** for spatial analysis.
* Deploy on **Streamlit Cloud / Heroku / AWS** for global accessibility.

---

## 👩‍💻 Team

| Name                        | PRN       | Role                                    |
| --------------------------- | --------- | --------------------------------------- |
| **Aniket Ashokrao Gawande** | 124B1F024 | Data Processing & Dashboard Development |
| **Atharv Sampat Shinde**    | 124B1F029 | Machine Learning & Visualization        |
| **Ashutosh Sanjay More**    | 124B1F036 | Backend Integration & Reporting         |

**Guided by:** *Mrs. Radha Deoghare Ma’am — Course Coordinator*
**Department of Information Technology, PCCOE**

---

## 📜 References

* [COVID19India Data Repository](https://data.covid19india.org)
* [Ministry of Health and Family Welfare (MOHFW)](https://www.mohfw.gov.in/)
* [CoWIN Portal](https://www.cowin.gov.in/)
* [Facebook Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start)
* [Statsmodels ARIMA Docs](https://www.statsmodels.org/stable/)
* [Scikit-Learn Documentation](https://scikit-learn.org/stable/)

---

## 🏁 Conclusion

This project demonstrates the **end-to-end application of data science** — from raw data collection to actionable forecasting — using **modern visualization and machine learning tools**.
By combining analytical rigor with intuitive design, the **COVID-19 India Dashboard** stands as a powerful educational and decision-support tool during global health challenges.

---

### 💖 Made with Passion & Purpose

> “Turning Data into Decisions — One Dashboard at a Time.”

```

---
I can generate that next for your README visual polish.
```
