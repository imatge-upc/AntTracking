
from collections import defaultdict
from docopt import docopt
import numpy as np
import pandas as pd
import pathlib
import sys


def acumulate_probs(probs):
    return [sum(probs[:i]) for i in range(len(probs))] + [1]

def split_pandas(pandas_table, probs, verbose=False):
    perm = np.random.permutation(len(pandas_table.index)) # List would be without .index

    try:
        for _ in probs : break
    except:
        probs = [probs]
    
    probs = acumulate_probs(probs)

    splits = []
    len_ = len(perm)
    for p0, p1 in zip(probs[:-1], probs[1:]):
        if verbose : print(f"{int(len_ * p0)} ---- {int(len_ * p1)}")
        splits.append(pandas_table.iloc[perm[int(len_ * p0) : int(len_ * p1)]]) # List would be without .iloc

    return tuple(splits)


def main(ann_path, outputs_root, probs, names, verbose=True):
    
    ann_table = pd.read_csv(ann_path, delimiter=',', header=None)
    colors = ann_table[1].unique()
    ann_color_splits= defaultdict()
    for color in colors:
        ann_color_splits[color] = split_pandas(ann_table[ann_table[1] == color], probs=probs, verbose=verbose)
    ann_splits = zip(*ann_color_splits.values())
    pathlib.Path(outputs_root).mkdir(parents=True, exist_ok=True)
    for l, n in zip(ann_splits, names):
        pd.concat(l).to_csv(f'{outputs_root}{n}', header=False, index=False)

DOCTEXT = f"""
Usage:
  split_per_color_dataset.py <ann_path> [options] (--probs=<p>... --names=<n>...)
  split_per_color_dataset.py -h | --help

Options:
  -h --help                             Show this screen. 
  --probs=<p>                           Float vector. Positive numbers that represent a fraction of the data available, they will be normalized (divided by their sum).
  --names=<n>                           Str vector. Name associated to each probs value, also the list output name.
  --outputs_root=<or>                   Str. Path where the list will be generated [default: ./].
  --verbose=<v>                         Bool. Some list creation script may show information as it runs, this parameter manage it [default: True].

"""

if __name__ == "__main__":
    opts = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    ann_path = opts['<ann_path>']
    outputs_root = opts['--outputs_root']

    probs = [float(x) for x in opts['--probs']]
    probs = [x/sum(probs) for x in probs]
    names = opts['--names']

    verbose = opts['--verbose'] == "True"

    main(ann_path, outputs_root, probs, names, verbose)
