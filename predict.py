import os
import numpy as np
import tensorflow as tf

MODEL_PATH = os.getenv("MODEL_PATH", "./model/rock_classifier.keras")
TEST_IMAGE_PATH = os.getenv("TEST_IMAGE_PATH", "./images/test.jpg")
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT", 180))
IMG_WIDTH = int(os.getenv("IMG_WIDTH", 180))
CLASS_NAMES = os.getenv("CLASS_NAMES", "marked,unmarked").split(",")

model = tf.keras.models.load_model(MODEL_PATH)

img = tf.keras.utils.load_img(TEST_IMAGE_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    f"This image most likely belongs to {CLASS_NAMES[np.argmax(score)]} "
    f"with a {100 * np.max(score):.2f}% confidence."
)