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

# 4. Split dataset (IMPORTANT: stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Feature scaling (IMPORTANT)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 🔥 Check sizes (debug safe)
print("Length check:", len(y_test), len(y_pred))

# 8. Confusion Matrix (force 2x2)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print("\nConfusion Matrix:\n", cm)

# 9. Extract values
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

# 10. Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
