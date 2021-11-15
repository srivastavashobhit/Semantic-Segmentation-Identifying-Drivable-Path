import tensorflow as tf


def read_file(image_url, mask_url):
    image = tf.io.read_file(image_url)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # this also set the value between 0 and 1

    mask = tf.io.read_file(mask_url)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)

    return image, mask


def resize(image, mask):
    shape = (96, 128)
    image = tf.image.resize(image, shape, method='nearest')
    mask = tf.image.resize(mask, shape, method='nearest')

    return image, mask
