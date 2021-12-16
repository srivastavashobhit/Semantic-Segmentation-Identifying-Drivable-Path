import tensorflow as tf


def read_image(image_url):
    image = tf.io.read_file(image_url)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # this also set the value between 0 and 1

    return image


def read_mask(mask_url):
    mask = tf.io.read_file(mask_url)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return mask


def resize_image(image):
    shape = (96, 128)
    image = tf.image.resize(image, shape, method='nearest')

    return image


def resize_mask(mask):
    shape = (96, 128)
    mask = tf.image.resize(mask, shape, method='nearest')

    return mask


def read_image_mask(image_url, mask_url):
    return read_image(image_url), read_mask(mask_url)


def resize_image_mask(image, mask):
    return resize_image(image), resize_mask(mask)


def get_image_from_array(array):
    return tf.keras.preprocessing.image.array_to_img(array)
