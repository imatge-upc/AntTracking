
from docopt import docopt
import cv2 as cv
import numpy as np
import os
from pathlib import Path
import shutil
import sys


def ignore_filter(crop_path, verbose=True):

    keep_names = set()
    for subsed in os.listdir(crop_path): # train, val
        keep_names.update([Path(name).stem for name in os.listdir(os.path.join(crop_path, subsed))])
    
    def _ignore_patterns(path, names):

        wanted_dir = lambda name : os.path.isdir(os.path.join(path, name)) and (name not in ['crops'])
        ignore = set(name for name in names if (Path(name).stem not in keep_names) and (not wanted_dir(name)))

        if verbose:
            print(f'{len(ignore)} discarded{f" : {ignore}" if len(ignore) < 5 else ""}')

        return ignore
    
    return _ignore_patterns

DOCTEXT = f"""
Usage:
  validable_to_ultralytics.py <data_path> <output_name>

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    data_path = args['<data_path>']
    output_name = args['<output_name>']

    crop_path = os.path.join(data_path, 'crops/')

    output_path = os.path.join(os.path.dirname(os.path.normpath(data_path)), output_name)

    ignore = ignore_filter(crop_path)
    shutil.copytree(data_path, output_path, ignore=ignore)

    yaml_filename = os.path.join(output_path, f'{output_name}.yaml')

    config_text = f"""
    # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    path: ./{output_path}  # dataset root dir
    train: images/train  # train images (relative to 'path') 128 images
    val: images/val  # val images (relative to 'path') 128 images
    test:  # test images (optional)

    # Classes
    names:
        0: ant

    """

    with open(yaml_filename, 'w') as f:
        f.write(config_text)
