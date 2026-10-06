import pandas as pd
import plotly.express as px

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_used_cars.csv")


print("Dataset loaded successfully!")
print("Rows:", df.shape[0])

# --------------------------------------------------
# 1. Car Price Distribution
# --------------------------------------------------

fig1 = px.histogram(
    df,
    x="sale_price",
    nbins=50,
    title="Used Car Price Distribution",
    labels={"sale_price": "Sale Price"}
)

fig1.write_html("price_distribution.html")
print("1. Price distribution created!")

# --------------------------------------------------
# 2. Year vs Sale Price
# --------------------------------------------------

fig2 = px.scatter(
    df,
    x="yr_mfr",
    y="sale_price",
    title="Car Manufacturing Year vs Sale Price",
    labels={
        "yr_mfr": "Manufacturing Year",
        "sale_price": "Sale Price"
    },
    hover_data=["make", "model"]
)

fig2.write_html("year_vs_price.html")
print("2. Year vs price created!")

# --------------------------------------------------
# 3. Fuel Type vs Average Price
# --------------------------------------------------

fuel_price = (
    df.groupby("fuel_type")["sale_price"]
    .mean()
    .reset_index()
)

fig3 = px.bar(
    fuel_price,
    x="fuel_type",
    y="sale_price",
    title="Average Car Price by Fuel Type",
    labels={
        "fuel_type": "Fuel Type",
        "sale_price": "Average Sale Price"
    }
)

fig3.write_html("fuel_type_price.html")
print("3. Fuel type analysis created!")

# --------------------------------------------------
# 4. Transmission vs Average Price
# --------------------------------------------------

transmission_price = (
    df.groupby("transmission")["sale_price"]
    .mean()
    .reset_index()
)

fig4 = px.bar(
    transmission_price,
    x="transmission",
    y="sale_price",
    title="Average Car Price by Transmission",
    labels={
        "transmission": "Transmission",
        "sale_price": "Average Sale Price"
    }
)

fig4.write_html("transmission_price.html")
print("4. Transmission analysis created!")

# --------------------------------------------------
# 5. Top 10 Car Makes by Average Price
# --------------------------------------------------

make_price = (
    df.groupby("make")["sale_price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig5 = px.bar(
    make_price,
    x="make",
    y="sale_price",
    title="Top 10 Car Makes by Average Sale Price",
    labels={
        "make": "Car Make",
        "sale_price": "Average Sale Price"
    }
)

fig5.write_html("top_makes_price.html")
print("5. Top makes analysis created!")

# --------------------------------------------------
# 6. Actual vs Predicted Price
# --------------------------------------------------

predictions = pd.read_csv("dataset/model_predictions.csv")

fig6 = px.scatter(
    predictions,
    x="Actual Price",
    y="Predicted Price",
    title="Actual Price vs Predicted Price",
    labels={
        "Actual Price": "Actual Price",
        "Predicted Price": "Predicted Price"
    }
)

fig6.write_html("actual_vs_predicted.html")
print("6. Actual vs predicted chart created!")

print("\nAll Plotly visualizations created successfully!")