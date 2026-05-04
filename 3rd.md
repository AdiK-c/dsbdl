# 1. Import libraries
import pandas as pd
import numpy as np

# ------------------------------------------
# PART 1: Summary statistics grouped by categorical variable
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
data = pd.DataFrame({
    'Age_Group': ['Young', 'Young', 'Adult', 'Adult', 'Senior', 'Senior'],
    'Income': [25000, 30000, 40000, 45000, 50000, 55000]
})

print("Dataset:\n", data)

# Group by categorical variable and calculate statistics
grouped = data.groupby('Age_Group')['Income']

print("\nMean:\n", grouped.mean())
print("\nMedian:\n", grouped.median())
print("\nMinimum:\n", grouped.min())
print("\nMaximum:\n", grouped.max())
print("\nStandard Deviation:\n", grouped.std())

# ------------------------------------------
# PART 2: Iris dataset statistics
# ------------------------------------------

# Load iris dataset
iris = pd.read_csv("/home/adi/Desktop/DSBDL/Iris.csv")

print("\nFirst 5 rows of Iris dataset:\n", iris.head())

# Check column names (optional but good)
print("\nColumns:\n", iris.columns)

# Group by correct column name
group = iris.groupby('Species')  # ✅ FIX

# Mean
print("\nMean:\n", group.mean())

# Standard deviation
print("\nStandard Deviation:\n", group.std())

# Percentiles (25%, 50%, 75%)
print("\nPercentiles:\n", group.quantile([0.25, 0.5, 0.75]))
# Percentiles (25%, 50%, 75%)
print("\nPercentiles:\n", group.quantile([0.25, 0.5, 0.75]))
