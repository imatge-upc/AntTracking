from docopt import docopt
import sys
from reid_splited_tracks.reid_dist import reid_offline
import pandas as pd
import numpy as np
import os

DOCTEXT = f"""
Usage:
  join_tracks.py <split_tracks_file> <track_features_file> <output_file> [--thr=<th>] [--dist=<dist>]
"""

def create_dict(tracking_file):
        # Leer el archivo CSV
    df = pd.read_csv(tracking_file, header=None)
    column_names = ['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf', 'a', 'b', 'c']
    df.columns = column_names + [f'feature_{i}' for i in range(len(df.columns) - len(column_names))]
    df['feature_vector'] = df[df.columns[len(column_names):]].apply(lambda row: np.array(row), axis=1)
    df.drop(df.columns[len(column_names):-1], axis=1, inplace=True)
    
    track_features_dict= {}
    for index, row in df.iterrows():
        track_id = row['trackId']
        vector = row['feature_vector']

        if track_id in track_features_dict:
            track_features_dict[track_id].append(vector)
        else:
            track_features_dict[track_id] = [vector]
    return track_features_dict

if __name__=="__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    tracking_file = args['<track_features_file>']
    split_track_file = args['<split_tracks_file>']
    output_file = args['<output_file>']
    thr = float(args['--thr'])
    dist = float(args['--dist'])

    df_in = pd.read_csv(split_track_file)
    df_in.columns = ['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf', 'a', 'b', 'c']
    track_features = create_dict(tracking_file)

    reid_offline(df_in,track_features,thr,dist,output_file)

    
    

