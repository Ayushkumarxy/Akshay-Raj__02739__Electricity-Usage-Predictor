"""
Electricity Usage Predictorr
Project No.: 62

Loads past monthly electricity usage from a CSV file, fits a simple linear
regression line using NumPy, predicts the units consumed in a future month,
estimates the bill at Rs. 6 per unit, and plots the actual data with the trend line.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BILL_RATE = 6.0
CSV_FILE = "usage.csv"


def load_data(filename=CSV_FILE):
    """Load and validate the Month and Units columns from CSV (with or without header)."""
    try:
        raw_df = pd.read_csv(filename, header=None)
    except FileNotFoundError:
        raise ValueError(
            f"File '{filename}' was not found. Keep it in the same folder "
            "as this Python file."
        )
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")

    if raw_df.empty or raw_df.shape[1] < 2:
        raise ValueError("CSV must contain two columns: Month and Units.")

    # Check if the first row is a header (e.g., "Month,Units")
    first_val = str(raw_df.iloc[0, 0]).strip().lower()
    second_val = str(raw_df.iloc[0, 1]).strip().lower()
    if first_val == "month" or second_val == "units":
        data = raw_df.iloc[1:, :2].copy()
    else:
        data = raw_df.iloc[:, :2].copy()

    data.columns = ["Month", "Units"]

    # Convert columns to numeric values; invalid or missing entries become NaN
    data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
    data["Units"] = pd.to_numeric(data["Units"], errors="coerce")

    invalid_rows = data[["Month", "Units"]].isna().any(axis=1)
    if invalid_rows.any():
        bad_count = int(invalid_rows.sum())
        raise ValueError(
            f"Found {bad_count} missing or non-numeric row(s). "
            "Correct the Month and Units values in the CSV."
        )

    if (data["Month"] <= 0).any():
        raise ValueError("Month values must be positive numbers.")
    if (data["Units"] < 0).any():
        raise ValueError("Units cannot be negative.")

    data = data.sort_values("Month").reset_index(drop=True)
    return data


def fit_regression(data):
    """Fit Units = slope * Month + intercept and calculate R-squared."""
    if len(data) < 2:
        raise ValueError("At least 2 months of data are required for prediction.")

    x = data["Month"].to_numpy(dtype=float)
    y = data["Units"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    predicted_y = slope * x + intercept

    ss_res = np.sum((y - predicted_y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if np.isclose(ss_tot, 0):
        r2 = 1.0 if np.isclose(ss_res, 0) else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return float(slope), float(intercept), float(r2)


def predict_usage(month, slope, intercept):
    """Predict units and estimated bill for a requested month."""
    predicted_units = slope * month + intercept
    estimated_bill = predicted_units * BILL_RATE
    return float(predicted_units), float(estimated_bill)


def show_plot(data, slope, intercept, predicted_month=None, predicted_units=None):
    """Plot actual usage points and the fitted regression line."""
    x = data["Month"].to_numpy(dtype=float)
    y = data["Units"].to_numpy(dtype=float)

    max_x = max(x.max(), predicted_month) if predicted_month is not None else x.max()
    line_x = np.linspace(x.min(), max_x, 100)
    line_y = slope * line_x + intercept

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="blue", label="Actual Usage", marker="o")
    plt.plot(line_x, line_y, color="red", label="Regression Line (Predicted)")

    if predicted_month is not None and predicted_units is not None:
        plt.scatter(
            [predicted_month],
            [predicted_units],
            color="green",
            marker="x",
            s=100,
            label=f"Predicted (Month {predicted_month:g})",
        )

    plt.title("Actual vs Predicted")
    plt.xlabel("Month")
    plt.ylabel("Units")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    print("Line chart 'Actual vs Predicted' displayed")
    plt.show()


def main():
    print("=" * 55)
    print("           ELECTRICITY USAGE PREDICTOR")
    print("=" * 55)

    try:
        data = load_data()
        print("\nData Summary:")
        print(data.to_string(index=False))
        print(f"Total records: {len(data)} | Average usage: {data['Units'].mean():.2f} units")

        slope, intercept, r2 = fit_regression(data)

        print(f"\nSlope: {slope:.2f} | Intercept: {intercept:.2f} | R2: {r2:.2f}")

        while True:
            print("\nMenu:")
            print("1. Predict")
            print("2. Show Plot Only")
            print("3. Exit")

            choice = input("Enter your choice (1-3): ").strip()

            if choice == "1":
                month_text = input("Enter month to predict: ").strip()
                try:
                    month = float(month_text)
                    if month <= 0:
                        print("Please enter a positive month number.")
                        continue
                    units, bill = predict_usage(month, slope, intercept)
                    print(f"\nSlope: {slope:.2f} | Intercept: {intercept:.2f} | R2: {r2:.2f}")
                    print(f"Predicted units (Month {month:g}): {units:.2f}")
                    print(f"Estimated bill: Rs. {bill:.2f}")
                    show_plot(data, slope, intercept, month, units)
                except ValueError:
                    print("Invalid month. Please enter a numeric value.")

            elif choice == "2":
                show_plot(data, slope, intercept)

            elif choice == "3":
                print("Exiting Electricity Usage Predictor.")
                break

            else:
                print("Invalid choice. Please select 1, 2, or 3.")

    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()