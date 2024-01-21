from evaldet import Tracks
from evaldet.mot import MOTMetrics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from docopt import docopt
import sys 

DOCTEXT = f"""
Usage:
  scr_hota.py <track_MOTfile> 
"""
if __name__=="__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    tracking_file = args['<track_MOTfile>']
    pd.set_option('display.precision', 2)

    print(plt.rcParams["figure.figsize"])


    gt_tracks = Tracks.from_mot('/home/usuaris/imatge/pol.serra.i.montes/TFG/AntTracking/DATA/all_ants_0-007_gt_tcks_skip_overlaps.txt')

    eval_obj = MOTMetrics(clearmot_dist_threshold=0.5, id_dist_threshold=0.5)

    tracks = Tracks.from_mot(tracking_file)

    metrics = eval_obj.compute(gt_tracks, tracks, clearmot_metrics=True, id_metrics=True, hota_metrics=True)
    metrics['clearmot']['MOTP'] = 1 - metrics['clearmot']['MOTP']

    print('CLEARMOT')
    print(pd.DataFrame(metrics['clearmot'], index=['values']))
    print('-' * 10)
    print('IDs')
    print(pd.DataFrame(metrics['id'], index=['values']))
    print('-' * 10)
    print('HOTA')
    print(pd.DataFrame({k : v for k, v in metrics['hota'].items() if k in ['HOTA', 'DetA', 'AssA', 'LocA']}, index=['values']))