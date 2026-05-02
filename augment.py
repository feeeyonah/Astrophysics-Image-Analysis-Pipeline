import os
import random

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy import ndimage

# Configurating image directory
IMAGE_DIR = os.getenv("IMAGE_DIR", "./images")
IMAGE_NAME = os.getenv("IMAGE_NAME", "rock")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# Constants
# (Scorch mark)
MARK_A_MIN = int(os.getenv("MARK_A_MIN", 10))
MARK_A_MAX = int(os.getenv("MARK_A_MAX", 25))
MARK_B_MIN = int(os.getenv("MARK_B_MIN", 5))
MARK_B_MAX = int(os.getenv("MARK_B_MAX", 15))
MARK_OPACITY_MAX = float(os.getenv("MARK_OPACITY_MAX", 0.4))

# (Rotation and Zoom)
ROTATION_ANGLES = [int(x) for x in os.getenv("ROTATION_ANGLES", "0,180").split(",")]
ZOOM_MIN = float(os.getenv("ZOOM_MIN", 1.0))
ZOOM_MAX = float(os.getenv("ZOOM_MAX", 2.0))
CROP_SIZE = int(os.getenv("CROP_SIZE", 500))

# (Contrast)
BRIGHTNESS_MIN = int(os.getenv("BRIGHTNESS_MIN", 10))
BRIGHTNESS_MAX = int(os.getenv("BRIGHTNESS_MAX", 20))
DARKNESS_MIN = int(os.getenv("DARKNESS_MIN", 10))
DARKNESS_MAX = int(os.getenv("DARKNESS_MAX", 30))
PIXEL_MIN = int(os.getenv("PIXEL_MIN", 0))
PIXEL_MAX = int(os.getenv("PIXEL_MAX", 255))

# Image loading
def load_image(image_dir: str, image_name: str) -> np.ndarray:
    """Load a JPG image, converting from PNG first if necessary."""
    jpg_path = os.path.join(image_dir, f"{image_name}.jpg")
    png_path = os.path.join(image_dir, f"{image_name}.png")

    if not os.path.exists(jpg_path):
        if os.path.exists(png_path):
            img = plt.imread(png_path)
            plt.imsave(jpg_path, img)
            os.remove(png_path)
        else:
            raise FileNotFoundError(f"No image found at {jpg_path} or {png_path}")

    return plt.imread(jpg_path)

# Image augmentation functions
@jit
def mark(image: np.ndarray):
    """Creates an oval 'scorch mark' at a random location on the image."""
    a = np.random.randint(MARK_A_MIN, MARK_A_MAX)
    b = np.random.randint(MARK_B_MIN, MARK_B_MAX)
    ic = np.random.randint(a, image.shape[0] - a)
    jc = np.random.randint(b, image.shape[1] - b)
    opacity = np.random.uniform(0, MARK_OPACITY_MAX)

    for i in range(ic - b, ic + b):
        for j in range(jc - a, jc + a):
            di = i - ic
            dj = j - jc
            if np.sqrt((dj / a) ** 2 + (di / b) ** 2) < 1:
                image[i, j, :] = (image[i, j, :] * opacity)

    return image.astype(np.uint8)

def manipulator(img_input: np.ndarray):
    """Apply a random rotation, zoom, and crop to the image."""
    rot_angle = random.choice(ROTATION_ANGLES)
    rotated = ndimage.rotate(img_input, rot_angle, reshape=False)

    # Per-channel zoom
    scale = np.random.uniform(ZOOM_MIN, ZOOM_MAX, size=2)
    channels = [ndimage.zoom(rotated[:, :, c], scale) for c in range(3)]
    zoomed = np.stack(channels, axis=-1)

    # Cropping
    x = np.random.randint(0, zoomed.shape[0] - CROP_SIZE)
    y = np.random.randint(0, zoomed.shape[1] - CROP_SIZE)
    cropped = zoomed[x:x + CROP_SIZE, y:y + CROP_SIZE, :]

    return cropped.astype(np.uint8)

def contrast(img: np.ndarray):
    """Randomly brighten or darken the image."""
    if np.random.random() > CONTRAST_PROB:
        img = np.clip(img + np.random.randint(BRIGHTNESS_MIN, BRIGHTNESS_MAX), PIXEL_MIN, PIXEL_MAX)
    else:
        img = np.clip(img - np.random.randint(DARKNESS_MIN, DARKNESS_MAX), PIXEL_MIN, PIXEL_MAX)

    return img.astype(np.uint8)

# Main
def main():
    img = load_image(IMAGE_DIR, IMAGE_NAME)

    unmarked_dir = os.path.join(OUTPUT_DIR, "unmarked")
    marked_dir = os.path.join(OUTPUT_DIR, "marked")
    os.makedirs(unmarked_dir, exist_ok=True)
    os.makedirs(marked_dir, exist_ok=True)

    copies = int(input("How many copies?: "))

    for n in range(copies):
        augmented = contrast(manipulator(np.copy(img)))

        plt.imsave(os.path.join(unmarked_dir, f"unmarked_{IMAGE_NAME}_{n}.jpg"), augmented)
        plt.imsave(os.path.join(marked_dir, f"marked_{IMAGE_NAME}_{n}.jpg"), mark(np.copy(augmented)))

    print("Done.")

if __name__ == "__main__":
    main()
