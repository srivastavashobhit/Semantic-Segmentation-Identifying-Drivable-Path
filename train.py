from data_utils import *
from model_utils import *


def train_new_model():
    model = UNet(FILTERS, CLASSES, INPUT_SIZE)
    train_dataset, val_dataset = get_train_dataset(IMAGES_SRC, MASKS_SRC, VAL_SPLIT, BATCH_SIZE)
    print(model.summary())
    history = model.fit(x=train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=get_callbacks())
    return model, history


def train_from_ckpt():
    train_dataset, val_dataset = get_train_dataset(IMAGES_SRC, MASKS_SRC, VAL_SPLIT, BATCH_SIZE)
    model = get_model_from_checkpoint()
    print(model.summary)
    history = model.fit(x=train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=get_callbacks())
    return model, history


