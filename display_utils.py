import matplotlib.pyplot as plt
import tensorflow as tf


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask_one(predicted_mask):
    predicted_mask = tf.argmax(predicted_mask, axis=-1)
    predicted_mask = predicted_mask[..., tf.newaxis]
    return predicted_mask[0]


def create_mask(predicted_mask):
    predicted_mask = tf.argmax(predicted_mask, axis=-1)
    predicted_mask = predicted_mask[..., tf.newaxis]
    return predicted_mask


def display_inference(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
