import tensorflow as tf


class UNet(tf.keras.Model):

    @staticmethod
    def skip_connection(input1, input2):
        return tf.keras.layers.concatenate(
            inputs=[input1, input2],
            axis=3
        )

    def __init__(self, filters, classes, input_size):
        super(UNet, self).__init__()
        self.filters = filters
        self.classes = classes
        self.encoder_block_1_conv1, \
            self.encoder_block_1_conv2, \
            self.encoder_block_1_maxpool, \
            self.encoder_block_1_dropout1 = self.encoder_block(block_depth=0, dropout_rate=0.3)

        self.encoder_block_2_conv1, \
            self.encoder_block_2_conv2, \
            self.encoder_block_2_maxpool, \
            self.encoder_block_2_dropout2 = self.encoder_block(block_depth=1, dropout_rate=0.3)

        self.encoder_block_3_conv1, \
            self.encoder_block_3_conv2, \
            self.encoder_block_3_maxpool, \
            self.encoder_block_3_dropout3 = self.encoder_block(block_depth=2, dropout_rate=0.3)

        self.encoder_block_4_conv1, \
            self.encoder_block_4_conv2, \
            self.encoder_block_4_maxpool, \
            self.encoder_block_4_dropout4 = self.encoder_block(block_depth=3, dropout_rate=0.3)

        self.bottle_neck_block_conv1, \
            self.bottle_neck_block_conv2 = self.bottle_neck_block(block_depth=4)

        self.decoder_block_4_upconv, \
            self.decoder_block_4_conv1, \
            self.decoder_block_4_conv2 = self.decoder_block(block_depth=3)

        self.decoder_block_3_upconv, \
            self.decoder_block_3_conv1, \
            self.decoder_block_3_conv2 = self.decoder_block(block_depth=2)

        self.decoder_block_2_upconv, \
            self.decoder_block_2_conv1, \
            self.decoder_block_2_conv2 = self.decoder_block(block_depth=1)

        self.decoder_block_1_upconv, \
            self.decoder_block_1_conv1, \
            self.decoder_block_1_conv2 = self.decoder_block(block_depth=0)

        self.output_block_0_conv, \
            self.output_block_0_output, = self.output_block(classes=classes)

        self.build(input_size)
        self.summary()
        self.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    def encoder_block(self, block_depth, dropout_rate, kernel_size=3):
        filters = self.filters * (2 ** block_depth)

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )
        maxpool1 = tf.keras.layers.MaxPool2D(
            pool_size=2
        )

        dropout1 = tf.keras.layers.Dropout(
            rate=dropout_rate
        )

        return conv1, conv2, maxpool1, dropout1

    def bottle_neck_block(self, block_depth, kernel_size=3):
        filters = self.filters * (2 ** block_depth)

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        return conv1, conv2

    def decoder_block(self, block_depth, kernel_size=3):
        filters = self.filters * (2 ** block_depth)
        upconv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
        )

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )

        return upconv, conv1, conv2

    def output_block(self, classes=1, kernel_size=3):
        filters = self.filters

        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )

        output = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=1,
            padding='same'
        )

        return conv, output

    def call(self, inputs, training=False):

        x = self.encoder_block_1_conv1(inputs)
        x = self.encoder_block_1_conv2(x)
        skip_connection_input_1 = tf.identity(x)
        x = self.encoder_block_1_maxpool(x)
        if training:
            x = self.encoder_block_1_dropout1(x)

        x = self.encoder_block_2_conv1(x)
        x = self.encoder_block_2_conv2(x)
        skip_connection_input_2 = tf.identity(x)
        x = self.encoder_block_2_maxpool(x)
        if training:
            x = self.encoder_block_2_dropout2(x)

        x = self.encoder_block_3_conv1(x)
        x = self.encoder_block_3_conv2(x)
        skip_connection_input_3 = tf.identity(x)
        x = self.encoder_block_3_maxpool(x)
        if training:
            x = self.encoder_block_3_dropout3(x)

        x = self.encoder_block_4_conv1(x)
        x = self.encoder_block_4_conv2(x)
        skip_connection_input_4 = tf.identity(x)
        x = self.encoder_block_4_maxpool(x)
        if training:
            x = self.encoder_block_4_dropout4(x)

        x = self.bottle_neck_block_conv1(x)
        x = self.bottle_neck_block_conv2(x)

        x = self.decoder_block_4_upconv(x)
        x = UNet.skip_connection(x, skip_connection_input_4)
        x = self.decoder_block_4_conv1(x)
        x = self.decoder_block_4_conv2(x)

        x = self.decoder_block_3_upconv(x)
        x = UNet.skip_connection(x, skip_connection_input_3)
        x = self.decoder_block_3_conv1(x)
        x = self.decoder_block_3_conv2(x)

        x = self.decoder_block_2_upconv(x)
        x = UNet.skip_connection(x, skip_connection_input_2)
        x = self.decoder_block_2_conv1(x)
        x = self.decoder_block_2_conv2(x)

        x = self.decoder_block_1_upconv(x)
        x = UNet.skip_connection(x, skip_connection_input_1)
        x = self.decoder_block_1_conv1(x)
        x = self.decoder_block_1_conv2(x)

        x = self.output_block_0_conv(x)
        x = self.output_block_0_output(x)

        return x
