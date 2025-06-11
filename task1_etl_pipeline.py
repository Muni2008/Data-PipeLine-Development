# CodTech Internship - Task 1: ETL Pipeline (Python Script Version)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 📥 STEP 1: Extract
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Initial Data Sample:")
print(df.head())

# 🔧 STEP 2: Transform
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('target', axis=1))

# Create new DataFrame with scaled features
scaled_df = pd.DataFrame(scaled_features, columns=iris.feature_names)
scaled_df['target'] = df['target']

# Train-test split
X = scaled_df.drop('target', axis=1)
y = scaled_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTransformed Data Sample:")
print(scaled_df.head())

# 💾 STEP 3: Load
scaled_df.to_csv("iris_cleaned.csv", index=False)
print("\nCleaned data saved as 'iris_cleaned.csv'")
