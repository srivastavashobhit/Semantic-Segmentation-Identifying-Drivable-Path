import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger

from display_utils import create_mask
from model import UNet

EPOCHS = 40
VAL_SUB_SPLIT = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
VAL_SPLIT = 0.2

FILTERS = 32
CLASSES = 23
INPUT_SIZE = ([32, 96, 128, 3])
INF_INPUT_SIZE = (1, 96, 128, 3)

CKPT_DIR = './saved_model/'
TENSORBOARD_LOG_DIR = "logs"
LOGGER_DIR = 'training.log'
SAVE_WEIGHTS_ONLY = False


def get_callbacks():
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=CKPT_DIR,
                                                    save_weights_only=SAVE_WEIGHTS_ONLY,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     patience=3)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      patience=3)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOGGER_DIR)

    csv_logger = CSVLogger(LOGGER_DIR)

    callbacks = [checkpoint, reduce_lr, early_stopping, tensorboard, csv_logger]

    return callbacks


def get_model_from_checkpoint():
    model = UNet(FILTERS, CLASSES, INPUT_SIZE)
    latest = tf.train.latest_checkpoint(CKPT_DIR)
    model.load_weights(latest)
    return model


def generate_prediction(model, input_image):
    prediction = model.predict(input_image)
    return create_mask(prediction)
