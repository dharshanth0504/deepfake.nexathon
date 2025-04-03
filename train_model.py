import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess images
def load_images(data_path, label):
    images, labels = [], []
    for filename in os.listdir(data_path):
        img_path = os.path.join(data_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable images
        img = cv2.resize(img, (224, 224)) / 255.0
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load real and deepfake images
real_images, real_labels = load_images('dataset/real', label=0)
fake_images, fake_labels = load_images('dataset/fake', label=1)

# Merge datasets
X = np.concatenate((real_images, fake_images), axis=0)
y = np.concatenate((real_labels, fake_labels), axis=0)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

# Build Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=15, validation_data=(X_test, y_test))

# Save Model
model.save('models/deepfake_detection_model.h5')
print("Model trained and saved successfully!")

