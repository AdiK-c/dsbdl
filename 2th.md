# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Create Academic Performance Dataset
data = pd.DataFrame({
    'Student_ID': [1,2,3,4,5,6,7,8,9,10],
    'Maths': [78, 85, np.nan, 90, 56, 88, 76, np.nan, 95, 60],
    'Science': [80, 89, 76, np.nan, 60, 92, 70, 65, np.nan, 58],
    'English': [75, 85, 80, 88, np.nan, 90, 72, 68, 94, 60],
    'Attendance': [85, 90, 78, 92, 60, 88, 75, 70, 95, 65],
    'Study_Hours': [2, 3, 1, 4, 1, 3, 2, 2, 5, 1]
})

print("Original Data:\n", data)

# ------------------------------------------
# 1. Handle Missing Values & Inconsistencies
# ------------------------------------------

# Check missing values
print("\nMissing Values:\n", data.isnull().sum())

# Fill missing values using mean
data.fillna(data.mean(numeric_only=True), inplace=True)

print("\nAfter Handling Missing Values:\n", data)

# ------------------------------------------
# 2. Detect and Handle Outliers (IQR Method)
# ------------------------------------------

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Detect outliers
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
print("\nOutliers:\n", outliers)

# Remove outliers
data_clean = data[~outliers.any(axis=1)]

print("\nData after removing outliers:\n", data_clean)

# ------------------------------------------
# 3. Data Transformation
# ------------------------------------------

# Apply log transformation to reduce skewness
data_clean['Maths_log'] = np.log(data_clean['Maths'] + 1)

print("\nAfter Log Transformation:\n", data_clean[['Maths', 'Maths_log']])

# ------------------------------------------
# Visualization (Optional but useful)
# ------------------------------------------

sns.histplot(data_clean['Maths'])
plt.title("Maths Score Distribution")
plt.show()
