from data_utils import *
from model_utils import *


def train_new_model(images_src=IMAGES_SRC, masks_src=MASKS_SRC, val_split=VAL_SPLIT, batch_size=BATCH_SIZE,
                    ):
    model = UNet(FILTERS, CLASSES, INPUT_SIZE)
    train_dataset, val_dataset = get_train_dataset(images_src, masks_src, val_split, batch_size)
    print(model.summary())
    history = model.fit(x=train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=get_callbacks())
    return model, history


def train_from_ckpt(images_src=IMAGES_SRC, val_src=MASKS_SRC, val_split=VAL_SPLIT, batch_size=BATCH_SIZE):
    train_dataset, val_dataset = get_train_dataset(images_src, val_src, val_split, batch_size)
    model = get_model_from_checkpoint()
    print(model.summary)
    history = model.fit(x=train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=get_callbacks())
    return model, history
