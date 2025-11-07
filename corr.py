# correlation_analysis.py
 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
# Load the national time-series data
data = pd.read_csv("data/case_time_series.csv")
 
# Select relevant columns for correlation analysis
correlation_df = data[['Daily Confirmed', 'Daily Recovered', 'Daily Deceased']]
 
# Calculate the correlation matrix
correlation_matrix = correlation_df.corr()
 
print("--- Correlation Matrix ---")
print(correlation_matrix)
 
# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Daily COVID-19 Metrics')
plt.show()
 
