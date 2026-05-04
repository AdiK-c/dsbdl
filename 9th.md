# 1. Import libraries
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load dataset
data = sns.load_dataset('titanic')

print("First 5 rows:\n", data.head())

# 3. Boxplot (Age vs Gender with Survival)
sns.boxplot(x='sex', y='age', hue='survived', data=data)

plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")

plt.show()
