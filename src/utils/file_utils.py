import errno
import os
from datetime import datetime


def create_directory(root_path):
    path = os.path.join(root_path, str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return path
