# Cell 1: Import Libraries
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Cell 2: Load the preprocessed data from .joblib files
# Assuming you have 'train_data.joblib' and 'test_data.joblib' files

# Load train and test datasets
x_train, y_train = joblib.load('train_data.joblib')
x_test, y_test = joblib.load('test_data.joblib')

# Reshape the flattened data back to (30, 30, 3)
x_train = x_train.reshape(-1, 30, 30, 3)
x_test = x_test.reshape(-1, 30, 30, 3)

# Standardize the data
scaler = StandardScaler()

# Flattened data needs to be scaled before reshaping
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)

x_train_scaled_flat = scaler.fit_transform(x_train_flat)
x_test_scaled_flat = scaler.transform(x_test_flat)

# Reshape back to 30x30x3
x_train_scaled = x_train_scaled_flat.reshape(len(x_train), 30, 30, 3)
x_test_scaled = x_test_scaled_flat.reshape(len(x_test), 30, 30, 3)

# Cell 3: Define MLP Model
def create_mlp_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(43)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model

# Create and compile model
input_shape = (x_train_scaled.shape[1],)
mlp_model = create_mlp_model(input_shape)

# Cell 4: Train the Model
history = mlp_model.fit(x_train_scaled, y_train, epochs=30, validation_data=(x_test_scaled, y_test), batch_size=64)

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
test_loss, test_accuracy = mlp_model.evaluate(x_test_scaled, y_test, verbose=2)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# Cell 7: Predict on the test dataset and calculate detailed metrics
y_pred = mlp_model.predict(x_test_scaled)
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
