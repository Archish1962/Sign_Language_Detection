import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

# Dataset directory (with subfolders per class)
dataset_path = "landmark_dataset"

# Load data
X, y = [], []
class_names = sorted(os.listdir(dataset_path))
class_indices = {name: idx for idx, name in enumerate(class_names)}

for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    for file in os.listdir(class_folder):
        if file.endswith(".npy"):
            path = os.path.join(class_folder, file)
            vector = np.load(path)
            X.append(vector)
            y.append(class_indices[class_name])

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(class_names))

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Build model
model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

# Save model and class labels
os.makedirs("landmark_model", exist_ok=True)
model.save("landmark_model/landmark_model.h5")
with open("landmark_model/labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("Model trained and saved.")
