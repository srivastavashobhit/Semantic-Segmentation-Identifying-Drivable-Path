from model import *

input_size = ([32, 96, 128, 3])
model = UNet(64)
inputs = tf.keras.layers.Input(input_size)
print(inputs)
unet = model.build(input_size)
print(model.summary())