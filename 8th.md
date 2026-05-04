# 1. Import libraries
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load inbuilt Titanic dataset
data = sns.load_dataset('titanic')

print("First 5 rows:\n", data.head())

# ------------------------------------------
# 1. Find patterns using seaborn
# ------------------------------------------

# Survival count
sns.countplot(x='survived', data=data)
plt.title("Survival Count")
plt.show()

# Survival by gender
sns.countplot(x='sex', hue='survived', data=data)
plt.title("Survival based on Gender")
plt.show()

# Survival by class
sns.countplot(x='pclass', hue='survived', data=data)
plt.title("Survival based on Passenger Class")
plt.show()

# ------------------------------------------
# 2. Histogram of Fare
# ------------------------------------------

sns.histplot(data['fare'], bins=20)
plt.title("Distribution of Fare")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()
