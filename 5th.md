# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# 2. Load dataset
data = pd.read_csv("Social_Network_Ads.csv")

print("First 5 rows:\n", data.head())

# 3. Select features and target
X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 🔥 ADD THIS (IMPORTANT - SCALING)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Logistic Regression model
model = LogisticRegression(max_iter=200)  # increased iterations
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Extract values
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

# 8. Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
