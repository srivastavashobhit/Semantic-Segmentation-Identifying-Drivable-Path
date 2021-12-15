import tensorflow as tf

EPOCHS = 40
VAL_SUB_SPLIT = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
VAL_SPLIT = 0.2


FILTERS = 32
CLASSES = 23
INPUT_SIZE = ([32, 96, 128, 3])

images_source_url = "data/carla/images"
masks_source_url = "data/carla/masks"


def get_callbacks():
    checkpoint_filepath = './test_model/checkpoint'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                    save_weights_only=True,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     patience=3)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      patience=3)
    tensorboard = tf.keras.callbacks.TensorBoard()

    callbacks = [checkpoint, reduce_lr, early_stopping, tensorboard]

    return callbacks

