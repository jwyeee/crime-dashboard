from flask import Flask, request, jsonify, send_file
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json

app = Flask(__name__)

# Load dataset once
def load_data():
    df = pd.read_csv("crime_data.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # Ensure proper datetime format

    # Remove "All" from dataset if it exists
    df = df[~df["district"].str.lower().eq("all")]
    df = df[~df["type"].str.lower().eq("all")]
    
    return df

df = load_data()

@app.route("/")
def home():
    return send_file("page.html")

@app.route("/filters", methods=["GET"])
def get_filters():
    filters = {
        "states": sorted(df["state"].dropna().unique().tolist()),
        "districts": sorted(df["district"].dropna().unique().tolist()),
        "categories": sorted(df["category"].dropna().unique().tolist()),
        "types": sorted(df["type"].dropna().unique().tolist())
    }
    return jsonify(filters)

@app.route("/update_filters", methods=["GET"])
def update_filters():
    selected_state = request.args.get("state", "All")
    selected_category = request.args.get("category", "All")

    filtered_df = df.copy()

    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == selected_state]

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]

    # Ensure no missing values before sending response
    districts = sorted(filtered_df["district"].dropna().unique().tolist())
    types = sorted(filtered_df["type"].dropna().unique().tolist())

    return jsonify({"districts": districts, "types": types})

@app.route("/predict", methods=["POST"])
def predict():
    selected_state = request.form.get("state", "All")
    selected_district = request.form.get("district", "All")
    selected_category = request.form.get("category", "All")
    selected_type = request.form.get("type", "All")

    filtered_df = df.copy()
    
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == selected_state]
    if selected_district != "All":
        filtered_df = filtered_df[filtered_df["district"] == selected_district]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["type"] == selected_type]

    if filtered_df.empty:
        return jsonify({"error": "No data available for the selected filters"})

    # Prepare data for Prophet
    filtered_df = filtered_df.groupby("date")["crimes"].sum().reset_index()
    filtered_df = filtered_df.rename(columns={"date": "ds", "crimes": "y"})

    if len(filtered_df) < 5:
        return jsonify({"error": "Not enough data for prediction"})

    # Train Prophet model
    model = Prophet(
        changepoint_prior_scale=0.2,
        seasonality_prior_scale=10,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    model.add_seasonality(name="quarterly", period=365.25 / 4, fourier_order=8)
    model.fit(filtered_df)

    # Make future predictions
    future = model.make_future_dataframe(periods=365, freq="D")
    forecast = model.predict(future)

    # Model evaluation
    merged = forecast.set_index("ds").join(filtered_df.set_index("ds"), how="inner")
    y_actual = merged["y"].dropna()
    y_predicted = merged["yhat"].dropna()

    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100
    accuracy = 100 - mape

    # Yearly values (tallying crimes by year)
    merged["year"] = merged.index.year
    yearly_values = merged.groupby("year").agg({"y": "sum", "yhat": "sum"}).reset_index()

    forecast_json = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_json["ds"] = forecast_json["ds"].astype(str)  # Convert dates to string for JSON
    forecast_list = forecast_json.to_dict(orient="records")

    # Prepare JSON response
    response_data = {
        "accuracy": accuracy,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "yearly_values": yearly_values.to_dict(orient="records"),
        "forecast_data": forecast_list
    }

    # Save predictions as JSON file
    with open("crime_predictions.json", "w") as f:
        json.dump(response_data, f, indent=4)

    print("\n✅ Crime predictions saved to crime_predictions.json")

    # Return the response as a JSON
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
