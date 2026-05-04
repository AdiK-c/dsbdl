# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load dataset
data = pd.read_csv("HousingData.csv")

# Display dataset
print("First 5 rows:\n", data.head())

# Check missing values
print("\nMissing values:\n", data.isnull().sum())

# Fill missing values (numeric only)
data.fillna(data.mean(numeric_only=True), inplace=True)

# 3. Convert categorical data to numeric (IMPORTANT FIX)
data = pd.get_dummies(data, drop_first=True)

# 4. Define features and target
X = data.drop('price', axis=1)
y = data['price']

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# 9. Actual vs Predicted
result = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nActual vs Predicted:\n")
print(result.head())
