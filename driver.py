import glob
import os
import sys
import argparse

from tensorflow.python.ops.numpy_ops import np_config

from data_utils import get_inference_dataset
from image_utils import read_image, resize_image
from train import train_new_model
from model_utils import get_model_from_checkpoint, generate_prediction, INF_INPUT_SIZE
from display_utils import display_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True, help="Provide a task.")
    parser.add_argument('-m', '--multiple', type=bool, help="Is multiple prediction.")
    parser.add_argument('-s', '--source', type=str, help="Source folder of images.")
    parser.add_argument('-i', '--image_url', type=str, help="Provide a Image URL.")
    args = parser.parse_args()

    if args.task == "training":
        train_new_model().history("accuracy")
    elif args.task == "inference":
        model = get_model_from_checkpoint()
        if args.multiple:
            assert args.source is not None
            # path = os.path.join(args.source, "*.png")
            # image_files = glob.glob(path)
            inference_dataset = get_inference_dataset(args.source, batch_size=32)
            for image in inference_dataset:
                prediction = model.predict(image)
        else:
            assert args.image_url is not None
            np_config.enable_numpy_behavior()
            input_image = resize_image(read_image(args.image_url)).reshape(INF_INPUT_SIZE)
            prediction = generate_prediction(model, input_image)
            print("^^", type(prediction))
            display_inference([input_image[0], prediction])
    else:
        print("Naacho saare gee fad ke.")

# #
# # model, history = train_new_model()
# # model.summary()
# #
# #
# model = get_model_from_checkpoint()
# image_src = "realtest.jpeg"
#
# np_config.enable_numpy_behavior()
# input_image = resize_image(read_image(image_src)).reshape(INF_INPUT_SIZE)
#
# input_image, prediction = generate_prediction(model, input_image)
# display_real([input_image[0], prediction])
