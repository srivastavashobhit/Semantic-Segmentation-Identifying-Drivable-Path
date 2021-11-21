from model import *
from data_utils import *

EPOCHS = 40
val_subsplits = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32

filters = 32
classes = 23


images_source_url = "data/carla/images"
masks_source_url = "data/carla/masks"

dataset = get_dataset(images_source_url, masks_source_url)

dataset.batch(BATCH_SIZE)

train_dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


input_size = ([32, 96, 128, 3])
unet = UNet(filters, classes)
unet.build(input_size)
unet.summary()
unet.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

model_history = unet.fit(train_dataset, epochs=EPOCHS)

