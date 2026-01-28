import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

df = pd.read_csv('data.csv')

rows_before = len(df)
df = df.drop_duplicates()
rows_after = len(df)

missing_values_count = df.isnull().sum()

# Remove age group 10-18
df = df[~((df['Age'] >= 10) & (df['Age'] <= 18))]


# Map risk levels
risk_map = {'high risk': 2, 'mid risk': 1, 'low risk': 0}
df['RiskLevel'] = df['RiskLevel'].replace(risk_map)

# Separate features and target
X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

if __name__ == "__main__":
    
    print(df.describe())
    print(f"Removed {rows_before - rows_after} duplicate rows.")
    print("Missing values in each column:")
    print(missing_values_count) 
    print(f"Number of records after cleaning: {len(df)}")
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    print("Original training set class distribution:", Counter(y_train))
    print("Resampled training set class distribution:", Counter(y_train_resampled))
