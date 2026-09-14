from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Walmart.csv"
MODEL_PATH = BASE_DIR / "_model.pkl"

FEATURE_NAMES = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Day",
    "Month",
    "Year",
]

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def load_sales_data():
    """Parse Walmart.csv and extract Month / Year for monthly benchmarks."""
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    return df


def monthly_benchmarks_for_store(store, year, sales_df):
    """
    Average weekly sales for months 1–12, filtered by Store and Year.
    Missing months in that year fall back to the overall store mean.
    """
    empty = [0.0] * 12
    if sales_df is None or sales_df.empty:
        return empty

    store_df = sales_df[sales_df["Store"] == store]
    if store_df.empty:
        overall = float(sales_df["Weekly_Sales"].mean())
        return [overall] * 12

    store_mean = float(store_df["Weekly_Sales"].mean())
    year_df = store_df[store_df["Year"] == year]
    benchmarks = []
    for month_num in range(1, 13):
        month_sales = year_df[year_df["Month"] == month_num]["Weekly_Sales"]
        if month_sales.empty:
            benchmarks.append(store_mean)
        else:
            benchmarks.append(float(month_sales.mean()))
    return benchmarks


def money(value):
    return f"${value:,.2f}"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/forecast", methods=["GET"])
def forecast():
    return render_template("forecast.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        store = int(request.form["store"])
        holiday_flag = int(request.form["holiday_flag"])
        temperature = float(request.form["temperature"])
        fuel_price = float(request.form["fuel_price"])
        cpi = float(request.form["cpi"])
        unemployment = float(request.form["unemployment"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        year = int(request.form["year"])

        feature_values = np.array(
            [
                store,
                holiday_flag,
                temperature,
                fuel_price,
                cpi,
                unemployment,
                day,
                month,
                year,
            ],
            dtype=float,
        )
        predicted_sales = float(model.predict(feature_values.reshape(1, -1))[0])

        intercept = float(model.intercept_)
        coefficients = np.asarray(model.coef_, dtype=float)

        contributions = []
        contribution_labels = []
        contribution_values = []
        for name, value, coef in zip(FEATURE_NAMES, feature_values, coefficients):
            dollar_impact = float(value * coef)
            contributions.append(
                {
                    "name": name.replace("_", " "),
                    "value": value,
                    "coef": float(coef),
                    "contribution": dollar_impact,
                    "formatted_value": f"{value:,.4f}".rstrip("0").rstrip("."),
                    "formatted_coef": f"{coef:,.6f}",
                    "formatted_contribution": money(dollar_impact),
                    "positive": dollar_impact >= 0,
                }
            )
            contribution_labels.append(name.replace("_", " "))
            contribution_values.append(round(dollar_impact, 2))

        try:
            sales_df = load_sales_data()
        except Exception:
            sales_df = None

        monthly_benchmarks = [
            round(v, 2) for v in monthly_benchmarks_for_store(store, year, sales_df)
        ]

        return render_template(
            "result.html",
            error=None,
            prediction=money(predicted_sales),
            prediction_raw=round(predicted_sales, 2),
            intercept=intercept,
            intercept_formatted=money(intercept),
            contributions=contributions,
            contribution_labels=contribution_labels,
            contribution_values=contribution_values,
            monthly_benchmarks=monthly_benchmarks,
            store=store,
            month=month,
            year=year,
            holiday_flag=holiday_flag,
            benchmarks_available=sales_df is not None,
        )
    except Exception as exc:
        return render_template(
            "result.html",
            error=str(exc),
            prediction=None,
            prediction_raw=0,
            intercept=0,
            intercept_formatted="$0.00",
            contributions=[],
            contribution_labels=[],
            contribution_values=[],
            monthly_benchmarks=[0] * 12,
            store=None,
            month=1,
            year=None,
            holiday_flag=None,
            benchmarks_available=False,
        )


if __name__ == "__main__":
    app.run(debug=True)
