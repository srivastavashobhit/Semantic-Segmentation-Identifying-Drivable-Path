import glob
import os
import argparse

from tensorflow.python.ops.numpy_ops import np_config

from utils.data_utils import get_inference_dataset
from utils.file_utils import create_directory
from utils.image_utils import read_image, resize_image, get_image_from_array
from train import train_new_model, train_from_ckpt
from utils.model_utils import get_model_from_checkpoint, generate_prediction
from utils.display_utils import display_inference, create_mask, create_mask_one
from utils.values_utils import INF_INPUT_SIZE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True, help="Provide a task.")
    parser.add_argument('-n', '--new', type=bool, help="Is new training.")
    parser.add_argument('-m', '--multiple', type=bool, help="Is multiple prediction.")
    parser.add_argument('-f', '--source_folder', type=str, help="Source folder of images.")
    parser.add_argument('-i', '--image_url', type=str, help="Provide a Image URL.")
    parser.add_argument('-d', '--display', type=bool, help="Display the prediction.")
    parser.add_argument('-s', '--save', type=bool, help="Save the prediction.")
    parser.add_argument('-e', '--extension', type=str, help="Image Extension.")
    args = parser.parse_args()

    if args.task == "training":
        if args.new:
            train_new_model().history("accuracy")
        else:
            train_from_ckpt().history("accuracy")
    elif args.task == "inference":
        model = get_model_from_checkpoint()
        if args.multiple:
            assert args.source_folder is not None
            assert args.extension is not None
            destination_path = create_directory(args.source_folder)
            names = [os.path.basename(x) for x in glob.glob(os.path.join(args.source_folder, "*."+args.extension))]
            inference_dataset = get_inference_dataset(args.source_folder, batch_size=32)
            predicted_tensors = create_mask(generate_prediction(model, inference_dataset))
            c = 0
            for predicted_tensor in predicted_tensors:
                prediction_image = get_image_from_array(predicted_tensor)
                prediction_image.save(os.path.join(destination_path, str(names[c]) + "_output."+args.extension))
                c += 1
        else:
            assert args.image_url is not None
            assert args.extension is not None
            np_config.enable_numpy_behavior()
            input_tensor = resize_image(read_image(args.image_url)).reshape(INF_INPUT_SIZE)
            predicted_tensor = create_mask_one(generate_prediction(model, input_tensor))
            prediction_image = get_image_from_array(predicted_tensor)
            input_image = get_image_from_array(input_tensor[0])
            if args.save:
                destination_path = create_directory("./data/test/")
                input_image.save(os.path.join(destination_path, "input."+args.extension))
                prediction_image.save(os.path.join(destination_path, "output."+args.extension))
            if args.display:
                display_inference([input_image, prediction_image])
