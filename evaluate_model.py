# evaluate_model.py

import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Load your dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

X = data_dict['data']    # Entire dataset features
y = data_dict['labels']  # Entire dataset labels

# 2. Load your trained RandomForest model from 'model.p'
with open('model.p', 'rb') as f:
    saved_model = pickle.load(f)
model = saved_model['model']

# 3. Predict on the entire dataset
y_pred = model.predict(X)

# 4. Calculate metrics
acc = accuracy_score(y, y_pred)
print(f"Accuracy on entire dataset: {acc * 100:.2f}%")

# 5. Confusion Matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

# 6. Classification Report
print("Classification Report:")
print(classification_report(y, y_pred))
