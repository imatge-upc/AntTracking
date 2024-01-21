import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import torch


def get_avgs(tracklet):
    sum_feat = 0
    for n, feat in enumerate(tracklet):
        sum_feat += feat
    avg = sum_feat/(n+1)
    return avg

            

def tracklets_distance (tracklet1:np.ndarray, tracklet2:np.ndarray) -> float:
    device = 'cpu'
    aa = torch.unsqueeze(get_avgs(torch.from_numpy(np.array(tracklet1)).to(device)), dim=0)
    bb = torch.unsqueeze(get_avgs(torch.from_numpy(np.array(tracklet2)).to(device)), dim=0)

    # Euclidean distance
    dd = np.squeeze(torch.cdist(aa,bb).cpu().numpy(), axis =0)[0]

    return dd


def compare_tracklets_all (tracklet_feats: np.ndarray, gallery:Dict[int,np.ndarray]) -> np.ndarray:
    '''
    Compare a tracklet against a gallery of tracklets. Return an array with the distances between the tracklet and all the tracklets in the gallery
    Parameters:
      tracklet_feats: feature vectors for the current track
      gallery       : dictionary with already finished tracks. For each track id (key) the dictionary contains the feature vectors of the track
      d_thresh      : distance threshold. Above this distance, two tracklets cannot be considered part of the same track
    Return value:
      An array of floats with the distances between the tracklet and all the elements of the gallery
    '''
    distances = np.zeros(len(gallery))
    for ii, (track_id, gal_track_feats) in enumerate(gallery.items()):
        distances[ii] = tracklets_distance(tracklet_feats, gal_track_feats) # Only one row
    return distances
    

def trackinfo (df:pd.DataFrame) -> pd.DataFrame:
    '''
    For each track, store the info of the start and end frames of the track. Return a pandas DataFrame with this info
    Parameters: 
      df             : DataFrame with the tracking information
    Return value:
      DataFrame with the info about the starting and ending frames of each track
    '''
    # Number of rows in the table
    nrows = len(df)

    # list of unique tracks
    tracks = list(set(df['trackId']))

    # Create an empty table that will contain, for each track, in which frame it starts and in which it ends
    dftrack = pd.DataFrame(columns = ['trackId', 'frameIni', 'frameEnd'])

    # For each track ...
    for ii,track in enumerate(tracks):
        # Create a new dataframe with all the frames that contain this track
        dft = df.loc[df.loc[:,'trackId']==track]['frameId']

        # Frame where this track appears for the firt time
        track_ini = min(dft)

        # Last frame where the track appears
        track_end = max(dft)

        # Add a new row to the table
        dftrack.loc[ii] = [track, track_ini, track_end]

    return (dftrack)


def check_overlap(grup_candi, grup_curr, overlap):
    intervals1 = overlap[grup_candi]
    intervals2 = overlap[grup_curr]
    
    for interval1 in intervals1:
        for interval2 in intervals2:
            if interval1[0] <= interval2[1] and interval1[1] >= interval2[0]:
                return False  #not valid
    return True

def check_distance(id1_xy,id2_xy):
    vec_dist = id2_xy-id1_xy
    dist_norm = np.linalg.norm(vec_dist)/np.linalg.norm([2092,4000])
    return dist_norm

def reid_offline(df_in, tracks_feats,thr,dist, output_reid):
    """
    NEW VERSION -- NO MERGING
    """

    df = df_in.copy(deep=True)
    
    tinfo_old = trackinfo(df)
    tinfo = {}
    
    for i, row in tinfo_old.iterrows():
        tinfo[row['trackId']] = (row['frameIni'], row['frameEnd'])


    # compute distance matrix
  
    distances = np.zeros((len(tracks_feats),len(tracks_feats)))

    for n, tf in enumerate(tracks_feats):
        distances[n,:] = compare_tracklets_all(tracks_feats[tf], tracks_feats)
        
    for i in range(len(tracks_feats)):
        for j in range(i, len(tracks_feats)):
            distances[i, j] = float('inf')
    
    distances = torch.from_numpy(distances)
        
    # key: order counter, value: list of associated tracklets
    def_tracks = {}
    overlap = {}
    max_track = np.max(df['trackId']) +1

    #fem un diccionari ordenat amb les distancies per aixi
    #no haver de trobar l'element x mes petit cada vegada
    ordered_dists = {}
    for i in range(0, len(distances)):
        for j in range(0, len(distances)):
            if i > j:
                key = (distances[i,j], int(list(tracks_feats.keys())[i]),int(list(tracks_feats.keys())[j]))
                ordered_dists[key] = 1

    ordered_dists = dict(sorted(ordered_dists.items()))
    for val, id1,id2 in ordered_dists.keys():
        if val < thr:
            if id1 not in def_tracks:

                max_track += 1
                overlap[max_track] = []
                overlap[max_track].append(tinfo[id1])
                
                def_tracks[id1] = max_track


            if id2 not in def_tracks:

                max_track += 1
                overlap[max_track] = []
                overlap[max_track].append(tinfo[id2])
                
                def_tracks[id2] = max_track


            valid = check_overlap(def_tracks[id1],def_tracks[id2], overlap)

            if valid:
                id1_df = df_in[(df_in['frameId'] == tinfo[id1][0]) & (df_in['trackId'] == id1)]
                id2_df = df_in[(df_in['frameId'] == tinfo[id2][1]) & (df_in['trackId'] == id2)]
                id1_xy = id1_df[['tlx', 'tly']].to_numpy()
                id2_xy = id2_df[['tlx', 'tly']].to_numpy()
                if check_distance(id1_xy,id2_xy)<dist:
                    for elem in overlap[def_tracks[id2]]:
                        overlap[def_tracks[id1]].append(elem)

                    overlap[def_tracks[id2]] = overlap[def_tracks[id1]]

                    to_change = def_tracks[id2]
                    for k,v in def_tracks.items():
                        if v == to_change:
                            def_tracks[k] = def_tracks[id1]

        if val > thr:
            for trackId, count in def_tracks.items():
                df.loc[df['trackId']==trackId, 'trackId'] = count  
            df_save = df.iloc[:, :6]
            df_save['conf']=-1
            df_save['a']=-1
            df_save['b']=-1
            df_save['c']=-1
            df_save.to_csv(output_reid, index=False, header=False)
            
            return df_save

    for trackId, count in def_tracks.items():
        df.loc[df['trackId']==trackId, 'trackId'] = count
    df_save = df.iloc[:, :6]
    df_save['conf']=-1
    df_save['a']=-1
    df_save['b']=-1
    df_save['c']=-1
    
    df_save.to_csv(output_reid, index=False, header=False)
                
    return df_save



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