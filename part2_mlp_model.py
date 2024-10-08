# -*- coding: utf-8 -*-
"""Part2_MLP_Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Rv2DlOitzuqRAS0xWSY5swGyTK1VvvVx
"""

import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report
import joblib


# Step 1: Mount Google Drive to access the images
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Preprocess the images and load labels from Google Drive
# Update the path based on your My Drive structure
images_path = '/content/drive/MyDrive/CS306_2024/images/'

# Define the class names
class_names = ['stop', '55_speed', 'green_light', 'red_light', 'sheep']

# Lists to store images and labels
images = []
labels = []

# Load and preprocess images
for label in class_names:  # Use the folder names as labels
    folder_path = os.path.join(images_path, label)
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_path}")
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)

            # Skip Google Drive shortcuts (these paths contain '.shortcut-targets-by-id')
            if '/.shortcut-targets-by-id/' in image_path:
                print(f"Skipping shortcut: {image_file}")
                continue

            # Skip non-image files based on their extension
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {image_file}")
                continue

            try:
                image = Image.open(image_path).resize((200, 200))
                image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
                images.append(image)
                labels.append(class_names.index(label))  # Use the index of the class name as the label
            except UnidentifiedImageError:
                print(f"Skipping invalid image file: {image_file}")
            except Exception as e:
                print(f"Error processing file {image_file}: {e}")

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Step 3: Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Cell 3: Define MLP Model
def create_mlp_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(43)  # Assuming 43 classes for the output
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model

# Create and compile model
input_shape = (200, 200, 3)
mlp_model = create_mlp_model(input_shape)

# Cell 4: Train the Model
history = mlp_model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), batch_size=64)

# Save the trained model
joblib.dump(mlp_model, 'Wesley_MLP_trained_model.joblib')
print("Model saved as 'Wesley_MLP_trained_model.joblib'")

# Cell 5: Plot Training History
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Cell 6: Evaluate the model on the test dataset
test_loss, test_accuracy = mlp_model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cell 7: Predict on the test dataset and calculate detailed metrics
y_pred = mlp_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

# Print the metrics
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")