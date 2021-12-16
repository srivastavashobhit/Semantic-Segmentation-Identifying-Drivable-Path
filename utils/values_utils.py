"""This file contains all the method input values at a single place"""

"""Training Inputs"""
EPOCHS = 40
VAL_SUB_SPLIT = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
VAL_SPLIT = 0.2

"""Model Inputs"""
FILTERS = 32
CLASSES = 23
INPUT_SIZE = ([32, 96, 128, 3])
INF_INPUT_SIZE = (1, 96, 128, 3)

"Callbacks Inputs"
CKPT_DIR = './saved_model_Dec16/'
LAST_CKPT_DIR = '../saved_model/'
TENSORBOARD_LOG_DIR = "../logs_Dec16"
LOGGER_DIR = 'training_Dec16.log'
SAVE_WEIGHTS_ONLY = False

"Data Inputs"
IMAGES_SRC = "../data/carla/images"
MASKS_SRC = "../data/carla/masks"
