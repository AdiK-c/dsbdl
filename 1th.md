!pip install pandas numpy matplotlib seaborn scikit-learn nltk
# 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 2. Dataset source
# Dataset: Titanic Dataset
# Source: https://www.kaggle.com/c/titanic/data

# 3. Load dataset
data = pd.read_csv("titanic.csv")

# Display first 5 rows
print("First 5 rows:\n", data.head())

# 4. Data Preprocessing

# Check missing values
print("\nMissing values:\n", data.isnull().sum())

# Fill missing values (numeric columns with mean)
data.fillna({
    'Age': data['Age'].mean(),
    'Fare': data['Fare'].mean()
}, inplace=True)

# Fill categorical missing values
data.fillna({
    'Age': data['Age'].mean(),
    'Fare': data['Fare'].mean(),
    'Embarked': data['Embarked'].mode()[0]
}, inplace=True)

# Describe dataset
print("\nStatistical Summary:\n", data.describe())

# Data info (types and non-null count)
print("\nDataset Info:")
print(data.info())

# Dimensions of dataset
print("\nShape of dataset:", data.shape)

# 5. Data Formatting and Normalization

# Check data types
print("\nData Types:\n", data.dtypes)

# Convert data types if needed
data['Survived'] = data['Survived'].astype('int')
data['Pclass'] = data['Pclass'].astype('category')

# Normalization (on numeric columns)
scaler = MinMaxScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

print("\nAfter Normalization:\n", data[['Age', 'Fare']].head())

# 6. Convert categorical variables into numeric

# Convert using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

print("\nEncoded Data:\n", data_encoded.head())
