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
# To run a test, just put any value in the TEST_PHASE (see example below) to avoid any changes in main folder.
TEST_PHASE = "DEC162021"  # TEST_PHASE = "DEC162021"
CKPT_DIR = "./saved_model/"+TEST_PHASE
LAST_CKPT_DIR = "./saved_model_dir/saved_model/"
TENSORBOARD_LOG_DIR = "./tensorboard_logs_dir/logs"+TEST_PHASE
LOGGER_DIR = "./csv_logger_dir/training"+TEST_PHASE+".log"
SAVE_WEIGHTS_ONLY = False

"Data Inputs"
IMAGES_SRC = "./data/carla/images"
MASKS_SRC = "./data/carla/masks"
