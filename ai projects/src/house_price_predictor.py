import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# LOAD DATA
# ============================================================

def load_data(filename):

    print("\n" + "=" * 60)
    print("                 LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(filename)

    print("\nDataset loaded successfully!")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print("\n" + "=" * 60)
    print("                 CLEANING DATA")
    print("=" * 60)

    df = df.copy()

    # --------------------------------------------------------
    # Remove unwanted spaces from column names
    # --------------------------------------------------------

    df.columns = df.columns.str.strip()

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "Area",
        "Rooms",
        "Bathrooms",
        "Garage Cars"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Clean Price
    #
    # Example:
    # R$ 2.200.000
    # becomes:
    # 2200000
    # --------------------------------------------------------

    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace("R$", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values before cleaning:")

    print(
        df[
            [
                "Area",
                "Rooms",
                "Bathrooms",
                "Garage Cars",
                "Price"
            ]
        ].isnull().sum()
    )

    # Remove rows where target Price is missing
    df = df.dropna(
        subset=["Price"]
    )

    # Fill numeric missing values
    for column in numeric_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"\nDuplicates removed: "
        f"{before - after}"
    )

    print(
        f"Rows after cleaning: "
        f"{len(df)}"
    )

    print("\nCleaning completed!")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\n" + "=" * 60)
    print("                 PREPARING DATA")
    print("=" * 60)

    features = [
        "Area",
        "Rooms",
        "Bathrooms",
        "Garage Cars"
    ]

    target = "Price"

    X = df[features]

    y = df[target]

    print("\nFeatures:")

    for feature in features:

        print(
            f"- {feature}"
        )

    print("\nTarget:")
    print("- Price")

    print("\nFeature shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    return X, y


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\n" + "=" * 60)
    print("                TRAIN TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):

    print("\n" + "=" * 60)
    print("                 TRAINING MODEL")
    print("=" * 60)

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nLinear Regression model "
        "trained successfully!"
    )

    print("\nModel coefficients:")

    for feature, coefficient in zip(
        X_train.columns,
        model.coef_
    ):

        print(
            f"{feature}: "
            f"{coefficient:.2f}"
        )

    print(
        f"\nIntercept: "
        f"{model.intercept_:.2f}"
    )

    return model


# ============================================================
# PREDICTIONS
# ============================================================

def make_predictions(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    print("\n" + "=" * 60)
    print("                  PREDICTIONS")
    print("=" * 60)

    for index, predicted in zip(
        X_test.index,
        predictions
    ):

        actual = y_test.loc[index]

        print(
            f"\nHouse {index}"
        )

        print(
            f"Actual Price: "
            f"R$ {actual:,.2f}"
        )

        print(
            f"Predicted Price: "
            f"R$ {predicted:,.2f}"
        )

    return predictions


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("                MODEL EVALUATION")
    print("=" * 60)

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"\nMAE  : R$ {mae:,.2f}"
    )

    print(
        f"MSE  : {mse:,.2f}"
    )

    print(
        f"RMSE : R$ {rmse:,.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return (
        mae,
        mse,
        rmse,
        r2
    )


# ============================================================
# CUSTOM HOUSE PREDICTION
# ============================================================

def predict_custom_house(model):

    print("\n" + "=" * 60)
    print("              CUSTOM HOUSE PREDICTION")
    print("=" * 60)

    try:

        area = float(
            input(
                "\nEnter house area (sq ft): "
            )
        )

        rooms = float(
            input(
                "Enter number of rooms: "
            )
        )

        bathrooms = float(
            input(
                "Enter number of bathrooms: "
            )
        )

        garage = float(
            input(
                "Enter number of garage cars: "
            )
        )

        house = pd.DataFrame(
            {
                "Area": [area],
                "Rooms": [rooms],
                "Bathrooms": [bathrooms],
                "Garage Cars": [garage]
            }
        )

        prediction = model.predict(
            house
        )[0]

        prediction = max(
            0,
            prediction
        )

        print("\n" + "-" * 60)

        print(
            f"Predicted House Price: "
            f"R$ {prediction:,.2f}"
        )

        print("-" * 60)

        return prediction

    except ValueError:

        print(
            "\nInvalid input."
        )

        return None


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    models_dir
):

    model_file = (
        models_dir /
        "house_price_model.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        f"\nModel saved to:\n"
        f"{model_file}"
    )

    return model_file


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph(
    model,
    X_test,
    y_test,
    charts_dir
):

    print("\n" + "=" * 60)
    print("                 CREATING GRAPH")
    print("=" * 60)

    predictions = model.predict(
        X_test
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6
    )

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect Prediction"
    )

    plt.xlabel(
        "Actual Price"
    )

    plt.ylabel(
        "Predicted Price"
    )

    plt.title(
        "Actual vs Predicted House Prices"
    )

    plt.legend()

    plt.grid(True)

    chart_file = (
        charts_dir /
        "actual_vs_predicted.png"
    )

    plt.savefig(
        chart_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"\nGraph saved to:\n"
        f"{chart_file}"
    )

    return chart_file


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    df,
    mae,
    mse,
    rmse,
    r2,
    report_file
):

    print("\n" + "=" * 60)
    print("                GENERATING REPORT")
    print("=" * 60)

    report = []

    report.append(
        "HOUSE PRICE PREDICTION - ML REPORT"
    )

    report.append(
        "=" * 55
    )

    report.append(
        "\nDATASET INFORMATION"
    )

    report.append(
        f"Number of houses: {len(df)}"
    )

    report.append(
        f"Number of columns: {len(df.columns)}"
    )

    report.append(
        "\nFEATURES"
    )

    report.append(
        "- Area"
    )

    report.append(
        "- Rooms"
    )

    report.append(
        "- Bathrooms"
    )

    report.append(
        "- Garage Cars"
    )

    report.append(
        "\nTARGET"
    )

    report.append(
        "- Price"
    )

    report.append(
        "\nMODEL"
    )

    report.append(
        "Algorithm: Multiple Linear Regression"
    )

    report.append(
        "\nMODEL EVALUATION"
    )

    report.append(
        f"MAE: R$ {mae:,.2f}"
    )

    report.append(
        f"MSE: {mse:,.2f}"
    )

    report.append(
        f"RMSE: R$ {rmse:,.2f}"
    )

    report.append(
        f"R²: {r2:.4f}"
    )

    report.append(
        "\nDATASET STATISTICS"
    )

    report.append(
        f"Average Area: "
        f"{df['Area'].mean():.2f}"
    )

    report.append(
        f"Average Rooms: "
        f"{df['Rooms'].mean():.2f}"
    )

    report.append(
        f"Average Bathrooms: "
        f"{df['Bathrooms'].mean():.2f}"
    )

    report.append(
        f"Average Garage Cars: "
        f"{df['Garage Cars'].mean():.2f}"
    )

    report.append(
        f"Average Price: "
        f"R$ {df['Price'].mean():,.2f}"
    )

    highest = df.loc[
        df["Price"].idxmax()
    ]

    report.append(
        "\nMOST EXPENSIVE HOUSE"
    )

    report.append(
        f"Address: {highest['Address']}"
    )

    report.append(
        f"Price: R$ {highest['Price']:,.2f}"
    )

    report.append(
        "\nCONCLUSION"
    )

    report.append(
        "A Multiple Linear Regression model "
        "was trained to predict house prices "
        "using area, rooms, bathrooms, and "
        "garage capacity."
    )

    report_file.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print(
        f"\nReport saved to:\n"
        f"{report_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)

    print(
        "          AI HOUSE PRICE PREDICTOR"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # PROJECT DIRECTORY
    # --------------------------------------------------------

    base_dir = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    data_file = (
        base_dir /
        "data" /
        "house_prices.csv"
    )

    charts_dir = (
        base_dir /
        "charts"
    )

    models_dir = (
        base_dir /
        "models"
    )

    reports_dir = (
        base_dir /
        "reports"
    )

    # --------------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------------

    charts_dir.mkdir(
        exist_ok=True
    )

    models_dir.mkdir(
        exist_ok=True
    )

    reports_dir.mkdir(
        exist_ok=True
    )

    report_file = (
        reports_dir /
        "ml_report.txt"
    )

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not data_file.exists():

        print(
            "\nERROR: Dataset not found!"
        )

        print(
            f"\nExpected location:"
        )

        print(
            data_file
        )

        return

    # --------------------------------------------------------
    # STEP 1: LOAD
    # --------------------------------------------------------

    df = load_data(
        data_file
    )

    # --------------------------------------------------------
    # STEP 2: CLEAN
    # --------------------------------------------------------

    df = clean_data(
        df
    )

    # --------------------------------------------------------
    # STEP 3: PREPARE
    # --------------------------------------------------------

    X, y = prepare_data(
        df
    )

    # --------------------------------------------------------
    # STEP 4: SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # --------------------------------------------------------
    # STEP 5: TRAIN
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # STEP 6: PREDICT
    # --------------------------------------------------------

    make_predictions(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # STEP 7: EVALUATE
    # --------------------------------------------------------

    (
        mae,
        mse,
        rmse,
        r2
    ) = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # STEP 8: CUSTOM PREDICTION
    # --------------------------------------------------------

    predict_custom_house(
        model
    )

    # --------------------------------------------------------
    # STEP 9: SAVE MODEL
    # --------------------------------------------------------

    save_model(
        model,
        models_dir
    )

    # --------------------------------------------------------
    # STEP 10: GRAPH
    # --------------------------------------------------------

    create_graph(
        model,
        X_test,
        y_test,
        charts_dir
    )

    # --------------------------------------------------------
    # STEP 11: REPORT
    # --------------------------------------------------------

    generate_report(
        df,
        mae,
        mse,
        rmse,
        r2,
        report_file
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "              PROJECT COMPLETED"
    )

    print("=" * 60)

    print("\nGenerated:")

    print(
        "📊 Prediction graph"
    )

    print(
        "🤖 Trained ML model"
    )

    print(
        "📄 ML report"
    )

    print(
        "\n🎉 Day 8 completed successfully!"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
