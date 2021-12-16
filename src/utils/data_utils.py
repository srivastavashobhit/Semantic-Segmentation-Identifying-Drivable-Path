from utils.image_utils import *

import os
import tensorflow as tf

from glob import glob
from sklearn.model_selection import train_test_split


def get_file_url_list(url, file_format="png"):
    path = os.path.join(url, "*" + file_format)
    file_list = glob(path)

    return file_list


def get_train_dataset_files(images_source_url, masks_source_url, validation_split):
    images_list = get_file_url_list(images_source_url)
    mask_list = get_file_url_list(masks_source_url)

    x_train, x_test, y_train, y_test = train_test_split(images_list, mask_list,
                                                        test_size=validation_split, random_state=40)

    x_train = tf.constant(x_train)
    y_train = tf.constant(y_train)

    x_test = tf.constant(x_test)
    y_test = tf.constant(y_test)

    train_dataset_files = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset_files = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_dataset_files, test_dataset_files


def get_train_dataset(images_source_url, masks_source_url, validation_split=0.2, batch_size=32):
    train_dataset_files, val_dataset_files = get_train_dataset_files(images_source_url, masks_source_url,
                                                                     validation_split)

    train_dataset = train_dataset_files.map(read_image_mask)
    train_dataset = train_dataset.map(resize_image_mask)

    val_dataset = val_dataset_files.map(read_image_mask)
    val_dataset = val_dataset.map(resize_image_mask)

    train_dataset.batch(batch_size)
    val_dataset.batch(batch_size)

    train_dataset = train_dataset.cache().batch(batch_size)
    val_dataset = val_dataset.cache().batch(batch_size)

    return train_dataset, val_dataset


def get_inference_dataset_files(images_source_url):
    images_list = get_file_url_list(images_source_url)

    images_list = tf.constant(images_list)

    inference_dataset_files = tf.data.Dataset.from_tensor_slices(images_list)

    return inference_dataset_files


def get_inference_dataset(images_source_url, batch_size=32):
    inference_files = get_inference_dataset_files(images_source_url)

    inference_dataset = inference_files.map(read_image)
    inference_dataset = inference_dataset.map(resize_image)

    inference_dataset.batch(batch_size)

    inference_dataset = inference_dataset.cache().batch(batch_size)

    return inference_dataset
