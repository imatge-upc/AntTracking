import pandas as pd
import torch
import torchvision
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import time


def split(input_path, output_path, mask, no_short_tracklet):
    
    tracking_file = input_path
    thr = 0.2
    
    # polyon needs to be changed for each different field, this one is for the 20221126[Badalona]CJB_vs_SESE_monocamera_50fps
    field_poly = mask
    
    old_df = pd.read_csv(tracking_file, header=None, names=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf', 'a', 'b', 'c'])
    
    df = old_df.copy(deep=True)
    
    frames = list(set(df['frameId']))
        
    already_seen = dict.fromkeys(df['trackId'], True)
        
    for i,frame in tqdm(enumerate(frames)):
        
        # find max trackID
        max_track = np.max(df['trackId']) +1
        
        #get all bbs of the same frame
        dff = df.loc[df.loc[:,'frameId']==frame][['trackId','tlx','tly','width','height']]
        
        start_time = time.time()
        if frame > 1:
            # check that there's no other tracklet with the same id and in that case, give it another id
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
        new = torch.cat((x1,y1,x2,y2), dim=1)
        crowd = torchvision.ops.box_iou(new,new).cuda()

        # check the occlusions in the frame
        row_check = (crowd <= thr)
        diagonal_indices = torch.arange(row_check.shape[0])

        # set diagonal elements to true because its between the same bb (true means no occlusion)
        row_check[diagonal_indices, diagonal_indices] = True
        
        # if the bb is outside the field set it to false
        i = -1   
        for index, row in dff.iterrows():
            x = row['tlx'] + row['width'] / 2 
            y = row['tly'] + row['height']
            #print(x, y, index)
            point = Point(x,y)
            i += 1
    
            if not field_poly.contains(point):
                row_check[i,i] = False
                #print('NOT IN FIELD: ', x, y)

        #eliminate the track id for the tracks that are in the conflictive zone
        false = torch.where(row_check == False)
        for pair in zip(false[0],false[1]):
            if df.loc[dff.index[0]+int(pair[0]),'trackId'] != -1:
                
                id1 = df.loc[dff.index[0]+int(pair[0]),'trackId']
                idx = df[(df['frameId'] > frame) & (df['trackId'] == id1)].index
                df.iloc[idx, df.columns.get_loc('trackId')] = max_track
                df.loc[dff.index[0]+int(pair[0]),'trackId'] = -1
                already_seen[max_track] = True
                max_track += 1
                
                
    
            if df.loc[dff.index[0]+int(pair[1]),'trackId'] != -1:
            
                id2 = df.loc[dff.index[0]+int(pair[1]),'trackId']
                idx = df[(df['frameId'] > frame) & (df['trackId'] == id2)].index
                df.iloc[idx, df.columns.get_loc('trackId')] = max_track
                df.loc[dff.index[0]+int(pair[1]),'trackId'] = -1
                already_seen[max_track] = True
                max_track += 1
                
    df = df[df['trackId'] != -1]
    
    if no_short_tracklet:
        v = df['trackId'].value_counts()
        df = df[df['trackId'].isin(v.index[v.gt(2)])]

    df.to_csv(output_path)
        
    return df


