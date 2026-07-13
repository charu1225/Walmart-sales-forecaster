# explore.py
import pandas as pd

# Load the dataset
df = pd.read_csv('Walmart.csv')

# 1. Print the first 5 rows to see what the data looks like
print("--- First 5 Rows ---")
print(df.head())

# 2. Check for missing (null) values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Look at basic statistical details (mean, min, max, etc.)
print("\n--- Data Statistics ---")
print(df.describe())
