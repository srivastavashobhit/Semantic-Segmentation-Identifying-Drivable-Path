import tensorflow as tf

from model import UNet

EPOCHS = 40
VAL_SUB_SPLIT = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
VAL_SPLIT = 0.2

FILTERS = 32
CLASSES = 23
INPUT_SIZE = ([32, 96, 128, 3])

IMAGES_SRC = "data/carla/images"
MASKS_SRC = "data/carla/masks"
CKPT_DIR = './test_model/checkpoint'


def get_callbacks():
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=CKPT_DIR,
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


def get_model_from_checkpoint():
    model = UNet(FILTERS, CLASSES, INPUT_SIZE)
    latest = tf.train.latest_checkpoint(CKPT_DIR)
    model.load_weights(latest)
    return model
