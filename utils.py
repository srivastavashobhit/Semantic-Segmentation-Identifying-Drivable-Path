import os
import re

from glob import glob


def get_batch_data_function(source_folder, input_image_shape):
    print(source_folder)

    def batch_data_function(batch_size):
        input_image_paths = glob(os.path.join(source_folder, "images", "*.png"))
        input_label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                             for path in glob(os.path.join(source_folder, 'gt_images', '*_road_*.png'))}
        print(input_image_paths)
        print(input_label_paths)

    return batch_data_function
