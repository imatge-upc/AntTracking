
from docopt import docopt
import json
import numpy as np
import pandas as pd
import sys


DOCTEXT = f"""
Usage:
  segment_detections.py <seq_path> <seg_path> <out_path>

"""


if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    seq_path = args['<seq_path>']
    seg_path = args['<seg_path>']
    out_path = args['<out_path>']

    #seq_dets = np.loadtxt(seq_path, delimiter=',')
    seq_dets = pd.read_csv(seq_path, header=0, names=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf','b','c', 'd'])

    table = pd.read_csv(seg_path, sep='\t', names=['ini', 'fin', 'out'])
    table['out'] = table['out'].apply(json.loads)

    indices_ok = table[['ini', 'fin']].apply(lambda x : np.arange(x['ini'], x['fin'] + 1), axis=1)
    indices_ok = np.hstack(indices_ok)

    outs = table['out'][table['out'].apply(lambda x : x != [])].explode('out').to_frame()['out'].to_list()
    indices_nok = pd.DataFrame(outs, columns=['ini', 'fin']).apply(lambda x : np.arange(x['ini'], x['fin'] + 1), axis=1)
    indices_nok = np.hstack(indices_nok)

    #seq_dets = seq_dets[np.isin(seq_dets[:, 0], indices_ok) & ~np.isin(seq_dets[:, 0], indices_nok), :]
    #np.savetxt(out_path, seq_dets.astype(int), fmt='%i', delimiter=',')

    seq_dets = seq_dets.loc[np.isin(seq_dets.loc[:, 0], indices_ok) & ~np.isin(seq_dets.loc[:, 0], indices_nok), :]
    seq_dets.to_csv(out_path, index=False, header=False)
