
from docopt import docopt
import json
import numpy as np
import os
import pandas as pd
import shutil
import sys


def mot_to_cvatmot(mot_table):
    cvatmot_table = mot_table[['fr_id', 'tck_id', 'x', 'y', 'w', 'h', 'not_ignore', 'cls_id', 'vis']].copy()
    cvatmot_table['cls_id'] = cvatmot_table['tck_id']
    cvatmot_table['vis'] = 1
    return cvatmot_table

def downsample(cvatmot_table):
    cvatmot_table['x'] = cvatmot_table['x'] // 2
    cvatmot_table['y'] = cvatmot_table['y'] // 2
    cvatmot_table['w'] = cvatmot_table['w'] // 2
    cvatmot_table['h'] = cvatmot_table['h'] // 2

def create_label(name):
    label = {
        "name": f"{int(name)}",
        "type": "rectangle",
        "attributes": []
    }
    return label


DOCTEXT = f"""
Usage:
  pred_to_cvat.py <seq_path> <output_file>
"""


if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    seq_path = args['<seq_path>']
    output_file = args['<output_file>']

    mot_table = pd.read_csv(seq_path, names=['fr_id', 'tck_id', 'x', 'y', 'w', 'h', 'not_ignore', 'cls_id', 'vis', 'NaN'])
    cvatmot_table = mot_to_cvatmot(mot_table)
    downsample(cvatmot_table)

    ids = cvatmot_table['cls_id'].unique()
    ids.sort()

    path = os.path.dirname(output_file)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    dir_name = os.path.join(path, base_name)

    os.makedirs(os.path.join(dir_name, 'gt'), exist_ok=False)
    cvatmot_table.to_csv(os.path.join(dir_name, 'gt', 'gt.txt'), index=False, header=False)
    np.savetxt(os.path.join(dir_name, 'gt', 'labels.txt'), ids, fmt='%u')

    shutil.make_archive(dir_name, 'zip', dir_name)
    shutil.rmtree(dir_name)

    labels = [create_label(n) for n in ids]
    with open(f'{dir_name}.json', 'w') as f:
        json.dump(labels, f)
