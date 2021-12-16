import os
import argparse

from tensorflow.python.ops.numpy_ops import np_config
from file_utils import create_directory
from image_utils import read_image, resize_image, get_image_from_array
from train import train_new_model
from model_utils import get_model_from_checkpoint, generate_prediction, INF_INPUT_SIZE
from display_utils import display_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True, help="Provide a task.")
    parser.add_argument('-m', '--multiple', type=bool, help="Is multiple prediction.")
    parser.add_argument('-f', '--source_folder', type=str, help="Source folder of images.")
    parser.add_argument('-i', '--image_url', type=str, help="Provide a Image URL.")
    parser.add_argument('-d', '--display', type=bool, help="Display the prediction.")
    parser.add_argument('-s', '--save', type=bool, help="Save the prediction.")
    args = parser.parse_args()

    if args.task == "training":
        train_new_model().history("accuracy")
    elif args.task == "inference":
        model = get_model_from_checkpoint()
        if args.multiple:
            pass
            # assert args.source_folder is not None
            # inference_dataset = get_inference_dataset(args.source_folder, batch_size=32)
            # destination_path = create_directory(args.source_folder)
            # name_counter = 0
            # for input_tensor in inference_dataset:
            #     predicted_tensor = model.predict(input_tensor)
            #     prediction_image = get_image_from_array(predicted_tensor)
            #     input_image = get_image_from_array(input_tensor[0])
            #     input_image.save(os.path.join(destination_path, str(name_counter) + "_input.png"))
            #     prediction_image.save(os.path.join(destination_path, str(name_counter) + "_output.png"))

        else:
            assert args.image_url is not None
            np_config.enable_numpy_behavior()
            input_tensor = resize_image(read_image(args.image_url)).reshape(INF_INPUT_SIZE)
            predicted_tensor = generate_prediction(model, input_tensor)
            prediction_image = get_image_from_array(predicted_tensor)
            input_image = get_image_from_array(input_tensor[0])
            if args.save:
                destination_path = create_directory("./data/carla/test/")
                input_image.save(os.path.join(destination_path, "input.png"))
                prediction_image.save(os.path.join(destination_path, "output.png"))
            if args.display:
                display_inference([input_image, prediction_image])

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
