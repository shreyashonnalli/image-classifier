import io
import numpy as np
from PIL import Image
import base64
import matplotlib.pyplot as plt
import tensorflow as tf


def resize_256_by_256(image):
    return tf.image.resize(image, size=[256, 256])


def color_correct(image):
    image = tf.image.adjust_contrast(image, contrast_factor=1.5)
    image = tf.image.adjust_brightness(image, delta=0.2)
    # image = tf.cast(image, tf.float32)/255.0
    return image


def rgb_to_ycbcr(rgb_image):
    R, G, B = tf.split(rgb_image, num_or_size_splits=3, axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128
    ycbcr_image = tf.stack([Y, Cb, Cr], axis=3)
    ycbcr_image = tf.squeeze(ycbcr_image, axis=-1)
    return ycbcr_image


def pipeline(image):
    image = resize_256_by_256(image)
    image = color_correct(image)
    image = rgb_to_ycbcr(image)
    image = image/255.0
    return image


file = open('demofile3.txt', 'r')
image = file.read()
file.close()
print(image)
image_decoded = base64.b64decode(image)
img = Image.open(io.BytesIO(image_decoded))
img = img.resize((256, 256))
# Convert the PIL image object to a NumPy array
np_img = np.array(img)
# Convert the image to 3 channels (if it is not already)
if np_img.ndim == 2:
    np_img = np.tile(np_img[:, :, np.newaxis], [1, 1, 3])

# Convert the image to RGB (if it is not already)
if np_img.shape[2] > 3:
    np_img = np_img[:, :, :3]
tf_img = np.expand_dims(np_img.astype(np.float32), axis=0)
tf_pipelined = pipeline(tf_img)
plt.imshow(tf_pipelined[0])
plt.show()
