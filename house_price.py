import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os
#Task 1: Dataset preparation

df= pd.read_csv("house_prices.csv")
print("Top 5 rows",df.head(5))
print ("Columans With Null Values", df.isna().sum())

#Task	2:	Data	preprocessing	

X=df[["area_sqft","bedrooms","bathrooms","house_age"]]
y=df["price"]

test_size=0.2
random_state=50

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

#Task	3:	Model	development	

# Createing the ML model instance
model = LinearRegression()

# Training the model using the training data X_train and y_train
model.fit(X_train, y_train)

#predicting modal output
y_pred=model.predict(X_test)


# Calculations of Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

#Task 4

# Task 4: Create the experiment
mlflow.set_experiment("House Price Prediction")

with mlflow.start_run(run_name="linear_regression_baseline"):
    #Model development
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    #Record model parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("features", list(X.columns))
    mlflow.log_params(model.get_params())  # fit_intercept, positive, etc.

    # Record evaluation metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    #Save the trained model artifact
    mlflow.sklearn.log_model(model, artifact_path="model")


   
    # Task	5:	Model	versioning	
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/house_price_model_v1.joblib")
    print("\nModel saved to models/house_price_model_v1.joblib")