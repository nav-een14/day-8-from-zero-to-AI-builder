# 🏠 Day 8 — AI House Price Predictor

A Machine Learning project that predicts house prices based on property features such as area, number of rooms, bathrooms, and garage capacity.

This project is part of my **30 Days of AI — From Zero to AI Builder** challenge.

## 🎯 Features

* Load real-world house price dataset
* Clean missing and invalid values
* Convert price values into numerical format
* Handle missing garage information
* Train a Linear Regression model
* Predict house prices
* Evaluate model performance
* Create actual vs predicted price graph
* Save trained ML model
* Allow custom house price prediction
* Generate ML performance report

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Git & GitHub

## 📊 Dataset

The dataset contains house information including:

* Address
* Area
* Rooms
* Bathrooms
* Garage Cars
* Price

The dataset contains more than **2,000 house records**.

## 📁 Project Structure

```text
Day-08-House-Price-Predictor/
│
├── data/
│   └── house_prices.csv
│
├── src/
│   └── house_price_predictor.py
│
├── charts/
│   └── actual_vs_predicted.png
│
├── models/
│   └── house_price_model.pkl
│
├── reports/
│   └── ml_report.txt
│
├── README.md
└── requirements.txt
```

## 🔄 Machine Learning Workflow

```text
House Price Dataset
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
Linear Regression
        ↓
Model Training
        ↓
Price Prediction
        ↓
Model Evaluation
        ↓
Save Model
```

## 🧠 Features Used

The model uses:

```text
Area
Rooms
Bathrooms
Garage Cars
```

Target:

```text
Price
```

## 📈 Model Evaluation

The model is evaluated using:

* MAE — Mean Absolute Error
* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error
* R² Score — Coefficient of Determination

## ▶️ How to Run

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

Run the project from the project folder:

```bash
python src\house_price_predictor.py
```

## 💻 Custom Prediction

The program asks for:

```text
Enter area (sqft):
Enter number of rooms:
Enter number of bathrooms:
Enter garage cars:
```

Example:

```text
Enter area (sqft): 1500
Enter number of rooms: 3
Enter number of bathrooms: 2
Enter garage cars: 1
```

The trained model then predicts the estimated house price.

## 🧠 Concepts Learned

* Supervised Machine Learning
* Regression
* Linear Regression
* Feature Selection
* Data Cleaning
* Missing Value Handling
* Train/Test Split
* Model Training
* Model Prediction
* MAE
* MSE
* RMSE
* R² Score
* Model Saving with Joblib

## 🚀 Future Improvements

* Try Random Forest Regression
* Try Gradient Boosting
* Add location-based features
* Improve feature engineering
* Compare multiple ML models
* Build a web interface using Flask/FastAPI
* Deploy the prediction system

## 📌 Day 8 Goal

Build a real-world **Machine Learning regression system** that can learn from house data and predict property prices.

**Day 8 Completed 🚀**
