
from docopt import docopt
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm


def make_maybe_fn_list(path):
    crop_path = os.path.join(path, 'images')
    fr_ids = [str(int(Path(filename).stem)) for filename in os.listdir(crop_path)]
    with open(os.path.join(path, 'maybe_fn.txt'), 'w') as f:
        f.write('\n'.join(fr_ids) + '\n')

def read_maybe(path):
    #  Return dict of frame_id with detections and valid yolo lines
    ids_path = os.path.join(path, f'{os.path.basename(path)}.txt')
    crops_path = os.path.join(path, 'crops')
    labels_path = os.path.join(path, 'labels')

    output = pd.DataFrame(columns=['fr', 'cls', 'x', 'y', 'w', 'h'])

    try:
        with open(ids_path) as f:
            frame_ids = [f'{int(id_):06}' for id_ in f] # f.readlines()
    except:
        return output
    
    get_line_number = lambda fname : int(fname.split("_")[-1].split(".")[0])
    for fr_id in tqdm(frame_ids):
        valid_lines = [get_line_number(fname) for fname in os.listdir(crops_path) if fname.startswith(fr_id)]
        label_path = [os.path.join(labels_path, f'{Path(fname).stem.split("_")[0]}.txt') for fname in os.listdir(crops_path) if fname.startswith(fr_id)]
        if len(label_path) == 0:
            continue
        with open(label_path[0], 'r') as f:
            lines = f.read().splitlines()
        for line_idx in valid_lines:
            values = lines[line_idx].split()
            row = {'fr' : int(fr_id), 'cls' : int(values[0]), 'x' : float(values[1]), 'y' : float(values[2]), 'w' : float(values[3]), 'h' : float(values[4])}
            output = output.append(row, ignore_index=True)

    return output

DOCTEXT = f"""
Usage:
  preanotated_to_mot.py <data_path> <output_path> <w> <h> [<seg_path>]

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    data_path = args['<data_path>']
    output_path = args['<output_path>']
    seg_path = args['<seg_path>']

    width = int(args['<w>'])
    height = int(args['<h>'])

    maybe_fn_path = os.path.join(data_path, 'maybe_fn')
    maybe_fp_path = os.path.join(data_path, 'maybe_fp')
    maybe_tp_path = os.path.join(data_path, 'maybe_tp')

    make_maybe_fn_list(maybe_fn_path)

    maybe_fn = read_maybe(maybe_fn_path)
    maybe_fp = read_maybe(maybe_fp_path)
    maybe_tp = read_maybe(maybe_tp_path)

    yolo_df = pd.concat([maybe_fn, maybe_fp, maybe_tp]).sort_values(by=['fr'], ignore_index=True)

    mot_df = pd.DataFrame()
    mot_df['fr'] = yolo_df['fr'].astype(int)
    mot_df['cls'] = 1
    mot_df['left'] = ((yolo_df['x'] - yolo_df['w'] / 2) * width).astype(int)
    mot_df['top'] = ((yolo_df['y'] - yolo_df['h'] / 2) * height).astype(int)
    mot_df['width'] = (yolo_df['w'] * width).astype(int)
    mot_df['height'] = (yolo_df['h'] * height).astype(int)
    mot_df['conf'] = float(1.)
    mot_df['x'] = -1
    mot_df['y'] = -1
    mot_df['z'] = -1

    if seg_path:
        table = pd.read_csv(seg_path, sep='\t', names=['ini', 'fin', 'out'])
        table['out'] = table['out'].apply(json.loads)

        indices_ok_ds = table[['ini', 'fin']].apply(lambda x : np.arange(x['ini'], x['fin'] + 1), axis=1)

        cls_ = list()
        for fr in mot_df['fr']:
            id_ = indices_ok_ds.apply(lambda x: fr in x)
            id_ = id_[id_ == True].index[0] + 1
            cls_.append(int(id_))

        mot_df['cls'] = cls_
        mot_df['cls'] = mot_df['cls'].astype(int)

    mot_df.to_csv(output_path, index=False, header=False)
