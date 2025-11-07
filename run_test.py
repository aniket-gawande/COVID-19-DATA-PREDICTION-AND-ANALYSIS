# runs_test_covid.py
 
from statsmodels.sandbox.stats.runs import runstest_1samp
import pandas as pd
 
# Load data
data = pd.read_csv("data/case_time_series.csv")
daily_cases = data['Daily Confirmed'].dropna()
 
# Perform the Runs Test
# We test if the sequence is random around its median
z_stat, p_value = runstest_1samp(daily_cases, correction=False)
 
print("--- Runs Test for Randomness of Daily Cases ---")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value}")
 
# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject the null hypothesis (p < 0.05).")
    print("The data sequence is not random and likely contains trends or patterns.")
else:
    print("\nConclusion: Fail to reject the null hypothesis (p >= 0.05).")
    print("There is no statistical evidence to suggest the data is not random.")
 
