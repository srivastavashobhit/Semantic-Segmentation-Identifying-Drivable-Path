from model import UNet
from data_utils import get_dataset
from model_utils import *

model = UNet(FILTERS, CLASSES, INPUT_SIZE)
train_dataset, val_dataset = get_dataset(images_source_url, masks_source_url, VAL_SPLIT, BATCH_SIZE)
model.summary()
history = model.fit(x=train_dataset,
                    epochs=EPOCHS,
                    validation_data=val_dataset,
                    callbacks=get_callbacks())
