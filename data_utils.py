from image_utils import *


import os
import tensorflow as tf

from glob import glob


def get_file_url_list(url, file_format="png"):
    path = os.path.join(url, "*" + file_format)
    file_list = glob(path)

    return file_list


def get_dataset_files(images_source_url, masks_source_url):
    image_list = get_file_url_list(images_source_url)
    mask_list = get_file_url_list(masks_source_url)
    image_filenames = tf.constant(image_list)
    mask_filenames = tf.constant(mask_list)
    dataset_files = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))

    return dataset_files


def get_dataset(images_source_url, masks_source_url):
    dataset = get_dataset_files(images_source_url, masks_source_url)
    dataset = dataset.map(read_file)
    dataset = dataset.map(resize)

    return dataset

