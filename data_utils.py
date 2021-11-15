import os
import tensorflow as tf

from glob import glob


def get_file_url_list(url, file_format="png"):
    path = os.path.join(url, "*" + file_format)
    file_list = glob(path)

    return file_list


def get_tf_dataset(image_list, mask_list):
    image_files = tf.constant(image_list)
    mask_files = tf.constant(mask_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))



