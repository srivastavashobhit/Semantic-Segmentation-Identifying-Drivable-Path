import os
import tensorflow as tf

from model import UNet
from data_utils import get_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCHS = 40
val_sub_splits = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32

filters = 32
classes = 23

images_source_url = "data/carla/images"
masks_source_url = "data/carla/masks"

train_dataset, test_dataset = get_dataset(images_source_url, masks_source_url, validation_split=0.2)

train_dataset.batch(BATCH_SIZE)
test_dataset.batch(BATCH_SIZE)

train_dataset = train_dataset.cache().batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

input_size = ([32, 96, 128, 3])
unet = UNet(filters, classes)
unet.build(input_size)
unet.summary()
unet.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

checkpoint_filepath = './saved_model/checkpoint'
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

model_history = unet.fit(x=train_dataset,
                         epochs=EPOCHS,
                         validation_data=test_dataset,
                         callbacks=callbacks)
