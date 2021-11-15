import tensorflow as tf
import numpy as np


class UNet(tf.keras.Model):

    def __init__(self, start_filters):
        super.__init__(self)
        self.start_filers = start_filters

    def encoder_block(self, inputs, block_depth, dropout_rate, kernel_size=3, training=False):
        filters = self.start_filers * (2 ** block_depth)

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )(inputs)

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )(conv1)
        maxpool1 = tf.keras.layers.MaxPool2D(
            pool_size=2
        )(conv2)

        if training:
            dropout1 = tf.keras.layers.Dropout(
                rate=dropout_rate
            )(maxpool1)
            return dropout1, conv2
        return maxpool1, conv2

    def bottle_neck_block(self, inputs, block_depth, kernel_size=3):
        filters = self.start_filers * (2 ** block_depth)

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )(inputs)

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )(conv1)

        return conv2

    def decoder_block(self, inputs, skip_connection_inputs, block_depth, kernel_size=3):
        filters = self.start_filers * (2 ** block_depth)
        upconv = tf.keras.layers.Conv2DTranpose(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
        )(inputs)

        skip_connection = tf.keras.layers.concatenate(
            inputs=[upconv, skip_connection_inputs],
            axis=3
        )

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )(skip_connection)

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )(conv1)

        return conv2

    def output_block(self, inputs, classes=1, kernel_size=3):
        filters = self.start_filers

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )(inputs)

        output = tf.keras.layers(
            filters=classes,
            kernel_size=1,
            padding='same'
        )(conv1)

        return output

    def call(self, inputs, training=False):
        encoder_output_1, encoder_skipp_connection_1 = self.encoder_block(
            inputs, block_depth=0, dropout_rate=0.1, training=training)

        encoder_output_2, encoder_skipp_connection_2 = self.encoder_block(
            encoder_output_1, block_depth=1, dropout_rate=0.1, training=training)

        encoder_output_3, encoder_skipp_connection_3 = self.encoder_block(
            encoder_output_2, block_depth=2, dropout_rate=0.1, training=training)

        encoder_output_4, encoder_skipp_connection_4 = self.encoder_block(
            encoder_output_3, block_depth=3, dropout_rate=0.3, training=training)

        bottle_neck_output = self.bottle_neck_block(
            inputs, block_depth=4)

        decoder_output_4 = self.decoder_block(
            bottle_neck_output, encoder_skipp_connection_4, block_depth=3)

        decoder_output_3 = self.decoder_block(
            decoder_output_4, encoder_skipp_connection_3, block_depth=2)

        decoder_output_2 = self.decoder_block(
            decoder_output_3, encoder_skipp_connection_2, block_depth=1)

        decoder_output_1 = self.decoder_block(
            decoder_output_2, encoder_skipp_connection_1, block_depth=0)

        outputs = self.output_block(
            decoder_output_1, classes=1)

        return outputs
