from image_utils import *

import os
import tensorflow as tf

from glob import glob
from sklearn.model_selection import train_test_split


def get_file_url_list(url, file_format="png"):
    path = os.path.join(url, "*" + file_format)
    file_list = glob(path)

    return file_list


def get_dataset_files(images_source_url, masks_source_url, validation_split):
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


def get_dataset(images_source_url, masks_source_url, validation_split=0.2, batch_size=32):
    train_dataset_files, val_dataset_files = get_dataset_files(images_source_url, masks_source_url, validation_split)

    train_dataset = train_dataset_files.map(read_file)
    train_dataset = train_dataset.map(resize)

    val_dataset = val_dataset_files.map(read_file)
    val_dataset = val_dataset.map(resize)

    train_dataset.batch(batch_size)
    val_dataset.batch(batch_size)

    train_dataset = train_dataset.cache().batch(batch_size)
    val_dataset = val_dataset.cache().batch(batch_size)

    return train_dataset, val_dataset
