import os
import re


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    if not os.path.isdir(directory):
        raise Exception("Dataset path '{}' does not exist.".format(directory))

    return sorted([os.path.join(root, f)
               for root, _, files in os.walk(directory) for f in files
               if re.match(r'^-1|\d', f)])
