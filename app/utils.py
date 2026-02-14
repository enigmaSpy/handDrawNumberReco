from PIL import Image
import numpy as np

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image