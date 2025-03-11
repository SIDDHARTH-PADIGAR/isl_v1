import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# 1. Ensure the "models" directory exists to save the trained model.
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Set key parameters for the model and data.
img_height, img_width = 64, 64  # Adjust as needed
num_channels = 3                # RGB images
num_classes = 28                # Adjust to match the number of classes in your dataset
batch_size = 32
epochs = 10

# 3. Build a simple CNN model from scratch using tf.keras.
model = tf.keras.models.Sequential([
    # Convolutional layer: 32 filters, 3x3 kernel.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(img_height, img_width, num_channels)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Second convolutional layer: 64 filters, 3x3 kernel.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the 2D feature maps to a 1D vector.
    tf.keras.layers.Flatten(),

    # Dense layer with 128 neurons.
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout to reduce overfitting.

    # Output layer: one neuron per class, using softmax for multi-class classification.
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 4. Compile the model with "adam" optimizer and "categorical_crossentropy" loss.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model architecture.
model.summary()

# 5. Prepare data with ImageDataGenerator (includes data augmentation).
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Path to your dataset directory (subfolders per class).
dataset_dir = './data/'

# Create training data generator (80% of data).
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Create validation data generator (20% of data), shuffle=False for consistent evaluation.
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 6. Train the model.
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 7. Evaluate the model on the validation set.
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 8. Generate a confusion matrix & classification report.
y_true = validation_generator.classes
y_pred_probs = model.predict(validation_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# 9. Save the trained model in the "models" directory.
model.save('models/asl_cnn_model.h5')
print("Model saved as 'models/asl_cnn_model.h5'")
