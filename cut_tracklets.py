
from docopt import docopt
import numpy as np
import pandas as pd
import sys
import torch
import torchvision


def cut_conflict(df, thr):
    frames = list(set(df['frameId']))
    already_seen = dict.fromkeys(df['trackId'], True)
    
    for i, frame in enumerate(frames):
        if i % 500 == 0 : print(frame)
        max_track = np.max(df['trackId']) + 1
        
        #get all bbs of the same frame
        dff = df.loc[df.loc[:, 'frameId'] == frame][['trackId', 'tlx', 'tly', 'width', 'height']]
        
        if frame > 1:
            #check that there's no other tracklet with the same id and in that case, give it another id
            anterior = set(df.loc[df.loc[:,'frameId']==frame-1]['trackId'])
            actual = set(df.loc[df.loc[:,'frameId']==frame]['trackId'])
        
            # if a ID was in the prior frame but not the current
            for value in (anterior - actual):
                #true means there's continuity 
                already_seen[value] = False

            # give new ID to the FALSE ones
            for val in actual:
                if already_seen[val] == False and val != -1:
                    df.loc[(df['trackId'] == val) & (df['frameId'] >= frame), 'trackId'] = max_track
                    already_seen[max_track] = True
                    max_track += 1
    
    
        # get coordinates
        d = torch.tensor(dff.values, dtype = torch.float)
        x1 = d[:, 1].clone().unsqueeze(1)
        y1 = d[:, 2].clone().unsqueeze(1)
        x2 = x1 + d[:, 3].clone().unsqueeze(1)
        y2 = y1+ d[:, 4].clone().unsqueeze(1)

        # box_iou is in format (x1,y1,x2,y2)
        new = torch.cat((x1, y1, x2, y2), dim=1)
        crowd = torchvision.ops.box_iou(new, new)#.cuda()

        # check the occlusions in the frame
        row_check = (crowd <= thr)
        diagonal_indices = torch.arange(row_check.shape[0])

        # set diagonal elements to true because its between the same bb (true means no occlusion)
        row_check[diagonal_indices, diagonal_indices] = True

        # eliminate the track d for the tracks that are in the conflictive zone (occlusion)
        false = torch.where(row_check == False) #false means occlusion and/or outside the field
        for pair in zip(false[0], false[1]):
            if df.loc[dff.index[0] + int(pair[0]), 'trackId'] != -1:
                
                id1 = df.loc[dff.index[0] + int(pair[0]), 'trackId']
                idx = df[(df['frameId'] > frame) & (df['trackId'] == id1)].index
                df.iloc[idx, df.columns.get_loc('trackId')] = max_track
                df.loc[dff.index[0] + int(pair[0]),'trackId'] = -1
                already_seen[max_track] = True
                max_track += 1
    
            if df.loc[dff.index[0] + int(pair[1]), 'trackId'] != -1:
                id2 = df.loc[dff.index[0] + int(pair[1]), 'trackId']
                idx = df[(df['frameId'] > frame) & (df['trackId'] == id2)].index
                df.iloc[idx, df.columns.get_loc('trackId')] = max_track
                df.loc[dff.index[0] + int(pair[1]), 'trackId'] = -1
                already_seen[max_track] = True
                max_track += 1

    return df      


DOCTEXT = f"""
Usage:
  video_to_apparences.py <tracking_file> <output_file> [--thr=<th>]

Options:
  --thr=<th>      IoU Thereshold: there is occlusion if 2 or more bboxes are overlapped more than thr. [default: 0.1]
"""


if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    tracking_file = args['<tracking_file>']
    output_file = args['<output_file>']
    thr = float(args['--thr'])

    seq_dets = np.loadtxt(tracking_file, delimiter=',', dtype=np.float64)
    df = pd.DataFrame(seq_dets[:, :10], columns=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf','a','b', 'c'])
    #feats = seq_dets[:, 10:]
    
    new_df = cut_conflict(df, thr)
    #good = new_df[new_df['trackId'] != -1]
    #good.to_csv(output_file, index=False, header=False)

    good = seq_dets
    good[:, 1] = new_df['trackId'] 
    good = good[new_df['trackId'] != -1]
    np.savetxt(output_file, good, delimiter=",", fmt='%.9g')
    
