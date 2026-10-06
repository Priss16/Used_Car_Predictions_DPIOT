import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load ML dataset
# --------------------------------------------------

df = pd.read_csv("dataset/ml_used_cars.csv")

print("ML dataset loaded successfully!")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop("sale_price", axis=1)
y = df["sale_price"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: sale_price")

# --------------------------------------------------
# 3. Identify numerical and categorical columns
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 5. Create Random Forest model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# --------------------------------------------------
# 6. Split data into training and testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

# --------------------------------------------------
# 9. Evaluate the model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# --------------------------------------------------
# 10. Display sample predictions
# --------------------------------------------------

results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(results)

# --------------------------------------------------
# 11. Save predictions
# --------------------------------------------------

results.to_csv(
    "dataset/model_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")

# --------------------------------------------------
# 12. Save trained model
# --------------------------------------------------

import joblib

joblib.dump(
    pipeline,
    "used_car_price_model.pkl"
)

print("Trained model saved successfully!")