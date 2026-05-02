import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Configuration
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "./output"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT", 180))
IMG_WIDTH = int(os.getenv("IMG_WIDTH", 180))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.3))
SEED = int(os.getenv("SEED", 123))
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS_INITIAL = int(os.getenv("EPOCHS_INITIAL", 15))
EPOCHS_FINAL = int(os.getenv("EPOCHS_FINAL", 100))
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", 0.2))
SHUFFLE_BUFFER = int(os.getenv("SHUFFLE_BUFFER", 1000))
CONV_FILTERS_1 = int(os.getenv("CONV_FILTERS_1", 16))
CONV_FILTERS_2 = int(os.getenv("CONV_FILTERS_2", 32))
CONV_FILTERS_3 = int(os.getenv("CONV_FILTERS_3", 64))
DENSE_UNITS = int(os.getenv("DENSE_UNITS", 128))
PIXEL_SCALE = float(os.getenv("PIXEL_SCALE", 1./255))


# Data loading
image_count = len(list(DATA_DIR.glob("*/*.jpg")))
unmarked = list(DATA_DIR.glob("Unmarked/*"))

print(f"Total images: {image_count}")
print(f"Unmarked images: {unmarked}")

PIL.Image.open(str(unmarked[0])).show()

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")


# Pipeline optimization
train_ds = train_ds.cache().shuffle(SHUFFLE_BUFFER).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalization check
normalization_layer = layers.Rescaling(PIXEL_SCALE)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(f"Pixel range: {np.min(first_image):.3f} - {np.max(first_image):.3f}")

num_classes = len(class_names)


# Initial model (no augmentation)
model = Sequential([
    layers.Rescaling(PIXEL_SCALE, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(CONV_FILTERS_1, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(CONV_FILTERS_2, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(CONV_FILTERS_3, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(DENSE_UNITS, activation='relu'),
    layers.Dense(num_classes),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_INITIAL,
)


# Training plots
epochs_range = range(EPOCHS_INITIAL)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Final model (with augmentation and dropout)
model = Sequential([
    data_augmentation,
    layers.Rescaling(PIXEL_SCALE),
    layers.Conv2D(CONV_FILTERS_1, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(CONV_FILTERS_2, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(CONV_FILTERS_3, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(DROPOUT_RATE),
    layers.Flatten(),
    layers.Dense(DENSE_UNITS, activation='relu'),
    layers.Dense(num_classes, name="outputs"),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINAL,
)

model.save(os.getenv("MODEL_PATH", "./model/rock_classifier.keras"))