
# NOTE: It seems useful for having just images and labels on a given folder, and having different train and val sets with that data (It should be a better practice than folder as sets)

from docopt import docopt
import os
from pathlib import Path
import random
import sys


DOCTEXT = f"""
    Usage:
      yolo_from_all_yolo_like.py <folder> <name> --p_val=<v> [--seed=<s>]
"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    input_ = args['<folder>']
    output = args['<name>']
    p_val = round(float(args['--p_val']), 2)
    assert (0 < p_val) and (p_val < 1)
    seed = int(args['--seed'] or 0) % 10000000000

    p_train = 1 - p_val
    random.seed(seed)

    name = Path(output).stem.split('.')[0]

    img_path = os.path.join(input_, 'images/')
    gt_path = os.path.join(input_, 'labels/')
    subsets_path = os.path.join(input_, f'subsets/{name}/')

    os.makedirs(subsets_path, exist_ok=True)

    train_filename = f'train_{int(p_train * 100):02}_seed{seed:010}.txt'
    val_filename = f'val_{int(p_val * 100):02}_seed{seed:010}.txt'

    train_path = os.path.join(subsets_path, train_filename)
    val_path = os.path.join(subsets_path, val_filename)
    dataset_path = os.path.join(input_, f'{name}.yaml')

    img_list = [os.path.abspath(os.path.join(img_path, img)) for img in os.listdir(img_path)]
    img_list = sorted(filter(lambda x : os.path.isfile(x), img_list)) # Sorted + Seed + Round to reproduce
    random.shuffle(img_list)

    cut_idx = int(len(img_list) * p_train)
    train_list = sorted(img_list[:cut_idx])
    val_list = sorted(img_list[cut_idx:])

    with open(train_path, 'w') as f:
        f.write('\n'.join(train_list))
    
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_list))

    with open(os.path.join(input_, 'classes.yaml'), 'r') as f:
        classes_str = f.read()

    dataset = f"""
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {os.path.basename(os.path.dirname(f'{input_}/'))} # dataset root dir
train: subsets/{name}/{train_filename} # train images (relative to 'path')
val: subsets/{name}/{val_filename} # val images (relative to 'path')
test:  # test images (optional)

{classes_str}
    """

    with open(dataset_path, 'w') as f:
        f.write(dataset)
