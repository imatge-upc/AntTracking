
from docopt import docopt
import numpy as np
import pandas as pd
import sys


def filter_large(df, max_side):
    new_df = df[(df['width'] <= max_side) & (df['height'] <= max_side)]
    return new_df


DOCTEXT = f"""
Usage:
  postprocess_track.py <tracking_file> <output_file> [--max_side=<ms>]

Options:
  --max_side=<ms>      Max height or width allowed fo a bbox. [default: 256]
"""


if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    tracking_file = args['<tracking_file>']
    output_file = args['<output_file>']
    max_side = int(args['--max_side'])

    df = pd.read_csv(tracking_file, header=0, names=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'a','b','c', 'd'])
    
    new_df = filter_large(df, max_side)
    
    new_df.to_csv(output_file, index=False, header=False)
    