# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 50)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 50)

# Load dataset
data = pd.read_csv("dataset/Iris.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Remove Id column
data = data.drop("Id", axis=1)

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display class distribution
print("\nFlower Species Count:")
print(data["Species"].value_counts())

# Correlation Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    data.drop("Species", axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.show()

# Data Visualization
sns.pairplot(data, hue="Species")
plt.show()

# Features and target
X = data.drop("Species", axis=1)
y = data["Species"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")

plt.show()

# Custom prediction
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(sample)

print("\nPredicted Species:")
print(prediction[0])