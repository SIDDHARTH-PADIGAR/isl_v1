import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # or you can keep using tf.keras.preprocessing if you like

# ---------------------------
# Step 1: Load the RandomForest model from pickle
# ---------------------------
with open('model.p', 'rb') as f:
    data = pickle.load(f)
model = data['model']  # This should be your trained RandomForestClassifier

# If you had a labels dictionary, load or define it too (optional).
# labels_dict = {0: 'A', 1: 'B', ...}  # Example if you have one

# ---------------------------
# Step 2: Preprocess the Image Exactly as in Training
# ---------------------------
img_path = './Data/9/hand1_j_bot_seg_1_cropped.jpeg'

# 1) Load the image
img = Image.open(img_path).convert('RGB')  # Convert to RGB if needed

# 2) Resize to match what you used during training (e.g., 64x64)
img = img.resize((64, 64))

# 3) Convert to a NumPy array
img_array = np.array(img)

# 4) Flatten the image into 1D (RandomForest expects a feature vector)
#    If your training pipeline used a different shape or color mode (grayscale), match that here.
img_flat = img_array.flatten()

# 5) If you normalized or scaled pixel values during training, do it now as well.
#    For example, if you divided by 255:
# img_flat = img_flat / 255.0

# ---------------------------
# Step 3: Make a Prediction
# ---------------------------
# RandomForest expects a 2D array: (num_samples, num_features)
# So we wrap `img_flat` in a list or use np.expand_dims
prediction = model.predict([img_flat])  # Returns a list or array of predicted labels
predicted_class = prediction[0]

print(f"Predicted class (numeric): {predicted_class}")

# If you have a labels_dict to map numeric classes to letters:
# print(f"Predicted sign: {labels_dict[predicted_class]}")

# ---------------------------
# Step 4: Display the Image with the Prediction
# ---------------------------
plt.imshow(img)
plt.title(f"Predicted class: {predicted_class}")
plt.axis('off')
plt.show()
