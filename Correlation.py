import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Preprocessing import X_train_resampled, X_test, y_train_resampled, y_test

# Compute correlation matrix on training features
corr_matrix = X_train_resampled.corr()

# Visualize correlation matrix (optional)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Find highly correlated features (absolute correlation > 0.4)
high_corr = np.where(abs(corr_matrix) > 0.4)
high_corr = [(corr_matrix.index[x], corr_matrix.columns[y]) 
             for x, y in zip(*high_corr) if x != y and x < y]

print("Highly correlated feature pairs (|corr| > 0.4):")
for pair in high_corr:
    print(pair)

# Optionally drop one feature from each highly correlated pair
to_drop = set([pair[1] for pair in high_corr])
X_train_reduced = X_train_resampled.drop(columns=to_drop)
X_test_reduced = X_test.drop(columns=to_drop, errors='ignore')  # ignore if feature not present
print(f"Dropped {len(to_drop)} features due to high correlation.")

# Initialize Logistic Regression
logreg = LogisticRegression(max_iter=5000, random_state=42)

# Train the model
logreg.fit(X_train_reduced, y_train_resampled)

# Make predictions on test set
y_pred = logreg.predict(X_test_reduced)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Mid', 'High'], 
            yticklabels=['Low', 'Mid', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
