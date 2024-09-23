# Cell 1: Import Libraries
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Cell 2: Load the preprocessed data from .joblib files
x_train, y_train = joblib.load('train_data.joblib')
x_test, y_test = joblib.load('test_data.joblib')

# Flatten the data if needed
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)

# Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Save the scaler for later use (optional)
joblib.dump(scaler, 'scaler_svm.joblib')

# Cell 3: Train the SVM model
# Use the best hyperparameters from the grid search
final_svm_model = SVC(C=100, gamma=0.0001, kernel='rbf', probability=True, max_iter=12000)

# Train the model
final_svm_model.fit(x_train_scaled, y_train)

# Save the trained model
joblib.dump(final_svm_model, 'Wesley_SVM_trained_model.joblib')
print("Final SVM model saved as 'Wesley_SVM_trained_model.joblib'.")

# Cell 4: Evaluate the SVM model on the test dataset
test_accuracy = final_svm_model.score(x_test_scaled, y_test)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# Cell 5: Performance Metrics on Test Set
y_pred = final_svm_model.predict(x_test_scaled)

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Cell 6: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: SVM Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Cell 7: Visualize Predictions (Optional)
# Assuming you want to visualize some predictions, similar to what was done in the MLP model
def plot_image(i, true_label, img, predicted_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.reshape(30, 30, 3))
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{predicted_label} ({true_label})", color=color)

# Visualize the first few predictions
plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plot_image(i, y_test[i], x_test[i], y_pred[i])

plt.tight_layout()
plt.show()
