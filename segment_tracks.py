# It does the same than segment_detections but ensuring 1 ant per frame and IDs from previous knowledge.

from docopt import docopt
import json
import numpy as np
import pandas as pd
import sys


DOCTEXT = f"""
Usage:
  segment_tracks.py <seq_path> <seg_path> <out_path>

"""


if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    seq_path = args['<seq_path>']
    seg_path = args['<seg_path>']
    out_path = args['<out_path>']

    seq_dets = np.loadtxt(seq_path, delimiter=',')

    table = pd.read_csv(seg_path, sep='\t', names=['ini', 'fin', 'out'])
    table['out'] = table['out'].apply(json.loads)

    indices_ok_ds = table[['ini', 'fin']].apply(lambda x : np.arange(x['ini'], x['fin'] + 1), axis=1)
    indices_ok = np.hstack(indices_ok_ds)

    outs = table['out'][table['out'].apply(lambda x : x != [])].explode('out').to_frame()['out'].to_list()
    indices_nok = pd.DataFrame(outs, columns=['ini', 'fin']).apply(lambda x : np.arange(x['ini'], x['fin'] + 1), axis=1)
    indices_nok = np.hstack(indices_nok)

    seq_dets = seq_dets[np.isin(seq_dets[:, 0], indices_ok) & ~np.isin(seq_dets[:, 0], indices_nok), :]

    valid_frame_numbers = np.unique(seq_dets[:, 0])
    if len(seq_dets[:, 0]) != len(valid_frame_numbers):
        times, _ = np.histogram(seq_dets[:, 0], bins=np.hstack([valid_frame_numbers, np.max(valid_frame_numbers) + 1]))
        indices_nok2 = valid_frame_numbers[times > 1]
        seq_dets = seq_dets[~np.isin(seq_dets[:, 0], indices_nok2)]

    for fr in seq_dets[:, 0]:
        id_ = indices_ok_ds.apply(lambda x: fr in x)
        id_ = id_[id_ == True].index[0] + 1
        seq_dets[seq_dets[:, 0] == fr, 1] = id_

    np.savetxt(out_path, seq_dets.astype(int), fmt='%i', delimiter=',')
