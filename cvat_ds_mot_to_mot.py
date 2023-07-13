
from docopt import docopt
import pandas as pd
import sys


DOCTEXT = f"""
Usage:
  cvat_ds_mot_to_mot.py <tracking_file> <output_file>
"""


if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    tracking_file = args['<tracking_file>']
    output_file = args['<output_file>']

    df = pd.read_csv(tracking_file, names=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf','a','b'])
    df['tlx'] = df['tlx'] * 2
    df['tly'] = df['tly'] * 2
    df['width'] = df['width'] * 2
    df['height'] = df['height'] * 2
    df['a'] = -1
    df['b'] = -1
    df['c'] = -1

    df.astype(int).to_csv(output_file, index=False, header=False)
