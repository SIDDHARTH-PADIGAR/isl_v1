import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load data from pickle file
print("Loading data...")
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict.get('data', [])
labels = data_dict.get('labels', [])

if not data or not labels:
    print("Error: No data or labels available for training.")
    exit(1)

# Verify feature count (should be 84 for two-hand approach)
if data and len(data[0]) != 84:
    print(f"Warning: Expected 84 features for two-handed ISL, but got {len(data[0])}.")
    user_input = input("Do you want to continue anyway? (y/n): ")
    if user_input.lower() != 'y':
        exit(1)
else:
    print(f"Confirmed: Using {len(data[0])} features per sample (two-hand approach).")

print(f"Dataset size: {len(data)} samples")
print(f"Number of classes: {len(set(labels))}")

# Check class distribution
unique_labels, counts = np.unique(labels, return_counts=True)
print("Class distribution:")
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

if len(x_train) == 0 or len(y_train) == 0:
    print("Error: Training set is empty after splitting.")
    exit(1)

print(f"Training set size: {len(x_train)}")
print(f"Testing set size: {len(x_test)}")

# Initialize and train the RandomForestClassifier
print("Training model...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
model.fit(x_train, y_train)

# Make predictions on the test set
print("Evaluating model...")
y_predict = model.predict(x_test)

# Calculate accuracy score
score = accuracy_score(y_predict, y_test)
print(f"Accuracy: {score * 100:.2f}%")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predict))

# Feature importance
print("\nTop 10 most important features:")
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]
for i in range(10):
    print(f"Feature {indices[i]}: {feature_importances[indices[i]]:.4f}")

# Save the trained model to a pickle file
print("Saving model...")
with open('model_84features.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'model_84features.p'")