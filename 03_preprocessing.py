import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_used_cars.csv")

print("Dataset loaded:", df.shape)

# Features we want to use for prediction
features = [
    "yr_mfr",
    "kms_run",
    "fuel_type",
    "city",
    "body_type",
    "transmission",
    "make",
    "model",
    "total_owners",
    "car_rating",
    "warranty_avail"
]

# Target variable
target = "sale_price"

# Create ML dataset
ml_data = df[features + [target]].copy()

print("\nSelected columns:")
print(ml_data.columns.tolist())

print("\nML dataset shape:")
print(ml_data.shape)

print("\nMissing values in selected columns:")
print(ml_data.isnull().sum())

# Save ML dataset
ml_data.to_csv("dataset/ml_used_cars.csv", index=False)

print("\nML dataset saved successfully!")