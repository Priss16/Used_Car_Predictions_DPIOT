import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Used_Car_Price_Prediction.csv")

print("=" * 50)
print("USED CAR DATASET ANALYSIS")
print("=" * 50)

# Dataset size
print("\n1. Dataset Shape")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Column names
print("\n2. Column Names")
for column in df.columns:
    print("-", column)

# Data types
print("\n3. Data Types")
print(df.dtypes)

# Missing values
print("\n4. Missing Values")
print(df.isnull().sum())

# Duplicate rows
print("\n5. Duplicate Rows")
print(df.duplicated().sum())

# Statistical information
print("\n6. Numerical Statistics")
print(df.describe())

# First 5 records
print("\n7. First 5 Records")
print(df.head())