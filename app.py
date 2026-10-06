from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import paho.mqtt.publish as mqtt_publish


# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = joblib.load("used_car_price_model.pkl")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/cleaned_used_cars.csv")
predictions_df = pd.read_csv("dataset/model_predictions.csv")


# ============================================================
# FIREBASE CONNECTION
# ============================================================

# IMPORTANT:
# Replace this filename with the exact name of your Firebase
# JSON file.

firebase_key = "autoluxe-ai-firebase-adminsdk-fbsvc-a577b2dcc6.json"

cred = credentials.Certificate(firebase_key)

firebase_admin.initialize_app(cred)

db = firestore.client()

print("Firebase connected successfully!")


# ============================================================
# HOME / PRICE PREDICTION
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    predicted_price = None

    if request.method == "POST":

        # ----------------------------------------------------
        # Get user input
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "yr_mfr":
                int(request.form["yr_mfr"]),

            "kms_run":
                int(request.form["kms_run"]),

            "fuel_type":
                request.form["fuel_type"],

            "city":
                request.form["city"],

            "body_type":
                request.form["body_type"],

            "transmission":
                request.form["transmission"],

            "make":
                request.form["make"],

            "model":
                request.form["model"],

            "total_owners":
                int(request.form["total_owners"]),

            "car_rating":
                request.form["car_rating"],

            # Convert string from HTML into Boolean
            "warranty_avail":
                request.form["warranty_avail"] == "True"

        }])


        # ----------------------------------------------------
        # Predict price
        # ----------------------------------------------------

        predicted_price = model.predict(input_data)[0]

        predicted_price = round(
            float(predicted_price),
            2
        )


        # ====================================================
        # SAVE PREDICTION TO FIREBASE
        # ====================================================

        prediction_data = {

            "yr_mfr":
                int(request.form["yr_mfr"]),

            "kms_run":
                int(request.form["kms_run"]),

            "fuel_type":
                request.form["fuel_type"],

            "city":
                request.form["city"],

            "body_type":
                request.form["body_type"],

            "transmission":
                request.form["transmission"],

            "make":
                request.form["make"],

            "model":
                request.form["model"],

            "total_owners":
                int(request.form["total_owners"]),

            "car_rating":
                request.form["car_rating"],

            "warranty_avail":
                request.form["warranty_avail"] == "True",

            "predicted_price":
                predicted_price,

            "timestamp":
                firestore.SERVER_TIMESTAMP
        }


        # Save into Firestore
        db.collection("predictions").add(
            prediction_data
        )


        print("Prediction saved to Firebase!")


    # ========================================================
    # DROPDOWN VALUES
    # ========================================================

    cities = sorted(
        df["city"].dropna().unique()
    )

    fuel_types = sorted(
        df["fuel_type"].dropna().unique()
    )

    body_types = sorted(
        df["body_type"].dropna().unique()
    )

    transmissions = sorted(
        df["transmission"].dropna().unique()
    )

    makes = sorted(
        df["make"].dropna().unique()
    )

    car_ratings = sorted(
        df["car_rating"]
        .dropna()
        .astype(str)
        .unique()
    )


    # ========================================================
    # SEND DATA TO HTML
    # ========================================================

    return render_template(

        "index.html",

        predicted_price=predicted_price,

        cities=cities,

        fuel_types=fuel_types,

        body_types=body_types,

        transmissions=transmissions,

        makes=makes,

        car_ratings=car_ratings
    )


# ============================================================
# PRICE PREDICTION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict_api():

    try:

        # ----------------------------------------------------
        # Get JSON data sent from frontend
        # ----------------------------------------------------

        data = request.get_json()


        # ----------------------------------------------------
        # Prepare input for ML model
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "yr_mfr":
                int(data["yr_mfr"]),

            "kms_run":
                int(data["kms_run"]),

            "fuel_type":
                data["fuel_type"],

            "city":
                data["city"],

            "body_type":
                data["body_type"],

            "transmission":
                data["transmission"],

            "make":
                data["make"],

            "model":
                data["model"],

            "total_owners":
                int(data["total_owners"]),

            "car_rating":
                data["car_rating"],

            "warranty_avail":
                data["warranty_avail"] == "True"

        }])


        # ----------------------------------------------------
        # Predict price using ML model
        # ----------------------------------------------------

        predicted_price = model.predict(input_data)[0]

        predicted_price = round(
            float(predicted_price),
            2
        )


        # ----------------------------------------------------
        # Save prediction to Firebase
        # ----------------------------------------------------

        prediction_data = {

            "yr_mfr":
                int(data["yr_mfr"]),

            "kms_run":
                int(data["kms_run"]),

            "fuel_type":
                data["fuel_type"],

            "city":
                data["city"],

            "body_type":
                data["body_type"],

            "transmission":
                data["transmission"],

            "make":
                data["make"],

            "model":
                data["model"],

            "total_owners":
                int(data["total_owners"]),

            "car_rating":
                data["car_rating"],

            "warranty_avail":
                data["warranty_avail"] == "True",

            "predicted_price":
                predicted_price,

            "timestamp":
                firestore.SERVER_TIMESTAMP
        }


        db.collection("predictions").add(
            prediction_data
        )


        print("Prediction API called successfully!")
        print("Prediction saved to Firebase!")


        # ====================================================
        # SEND PREDICTION TO MQTT
        # ====================================================

        mqtt_publish.single(
            "car/prediction/price",
            payload=str(predicted_price),
            hostname="localhost",
            port=1883
        )

        print("Prediction sent to MQTT!")


        # ----------------------------------------------------
        # Send result back to frontend
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "predicted_price":
                predicted_price,

            "message":
                "Prediction generated successfully"

        })


    except Exception as e:

        print("Prediction API error:", e)

        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 400


# ============================================================
# ANALYTICS API
# ============================================================

@app.route("/api/analytics")
def analytics_api():

    make = request.args.get(
        "make",
        "All"
    )

    fuel_type = request.args.get(
        "fuel_type",
        "All"
    )

    transmission = request.args.get(
        "transmission",
        "All"
    )


    # Start with complete dataset
    filtered = df.copy()


    # --------------------------------------------------------
    # Filter by Make
    # --------------------------------------------------------

    if make != "All":

        filtered = filtered[
            filtered["make"] == make
        ]


    # --------------------------------------------------------
    # Filter by Fuel
    # --------------------------------------------------------

    if fuel_type != "All":

        filtered = filtered[
            filtered["fuel_type"] == fuel_type
        ]


    # --------------------------------------------------------
    # Filter by Transmission
    # --------------------------------------------------------

    if transmission != "All":

        filtered = filtered[
            filtered["transmission"] == transmission
        ]


    # --------------------------------------------------------
    # Get prices
    # --------------------------------------------------------

    prices = (
        filtered["sale_price"]
        .dropna()
    )


    # --------------------------------------------------------
    # Send analytics data
    # --------------------------------------------------------

    return jsonify({

        "count":
            int(len(filtered)),

        "average_price":
            float(prices.mean())
            if len(prices)
            else None,

        "minimum_price":
            float(prices.min())
            if len(prices)
            else None,

        "maximum_price":
            float(prices.max())
            if len(prices)
            else None,

        "prices":
            prices.tolist(),

        "year":
            filtered["yr_mfr"]
            .fillna(0)
            .astype(int)
            .tolist(),

        "year_price":
            filtered["sale_price"]
            .fillna(0)
            .tolist(),

        "fuel_types":
            (
                filtered
                .groupby("fuel_type")["sale_price"]
                .mean()
                .dropna()
                .to_dict()
            ),

        "transmissions":
            (
                filtered
                .groupby("transmission")["sale_price"]
                .mean()
                .dropna()
                .to_dict()
            ),

        "top_makes":
            (
                filtered
                .groupby("make")["sale_price"]
                .mean()
                .dropna()
                .sort_values(ascending=False)
                .head(10)
                .to_dict()
            ),

        "actual_prices":
            predictions_df["Actual Price"].tolist(),

        "predicted_prices":
            predictions_df["Predicted Price"].tolist()
    })


# ============================================================
# PREDICTION HISTORY API
# ============================================================

@app.route("/api/predictions")
def predictions_api():

    try:

        # Get predictions from Firebase
        docs = (
            db.collection("predictions")
            .order_by(
                "timestamp",
                direction=firestore.Query.DESCENDING
            )
            .stream()
        )

        predictions = []


        for doc in docs:

            data = doc.to_dict()


            # Convert Firebase timestamp into readable text
            if data.get("timestamp"):

                data["timestamp"] = (
                    data["timestamp"]
                    .isoformat()
                )


            # Add document ID
            data["id"] = doc.id

            predictions.append(data)


        return jsonify({

            "success": True,

            "predictions":
                predictions
        })


    except Exception as e:

        print(
            "Prediction history API error:",
            e
        )

        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 400


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )