# ttest_waves_covid.py
 
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
 
# Load data
data = pd.read_csv("data/case_time_series.csv", parse_dates=['Date_YMD'])
data.set_index('Date_YMD', inplace=True)
 
# Define waves
wave1 = data['2020-08-01':'2020-11-30']
wave2 = data['2021-03-01':'2021-06-30']
 
# Calculate CFR, handling division by zero
wave1['CFR'] = (wave1['Daily Deceased'] / wave1['Daily Confirmed']).replace([np.inf, -np.inf], np.nan).dropna() * 100
wave2['CFR'] = (wave2['Daily Deceased'] / wave2['Daily Confirmed']).replace([np.inf, -np.inf], np.nan).dropna() * 100
 
# Perform the t-test
t_stat, p_value = ttest_ind(wave1['CFR'], wave2['CFR'], equal_var=False) # Welch's t-test
 
print("--- T-test for Case Fatality Rate (CFR) Between Waves ---")
print(f"Mean CFR Wave 1: {wave1['CFR'].mean():.2f}%")
print(f"Mean CFR Wave 2: {wave2['CFR'].mean():.2f}%")
print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_value}")
 
# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject the null hypothesis (p < 0.05).")
    print("There is a statistically significant difference in the CFR between Wave 1 and Wave 2.")
else:
    print("\nConclusion: Fail to reject the null hypothesis (p >= 0.05).")
