
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial import distance


def compute_ground_truth(trackIds, tracklets_df, gt_tracking_df, index):

    split_to_gt = dict()
    gt_to_split = defaultdict(list)

    for id_ in trackIds:

        query = tracklets_df[tracklets_df['trackId'] == id_].copy().reset_index().iloc[0]
        value_mask = np.all([gt_tracking_df[k] == query[k] for k in ['frameId', 'tlx', 'tly', 'width', 'height']], axis=0)

        assert sum([1 for v in value_mask if v]) == 1, id_

        key = index[id_]
        value = gt_tracking_df[value_mask]['trackId'].iloc[0]
        split_to_gt[key] = value
        gt_to_split[value].append(key)

    return split_to_gt, gt_to_split

def mean_descriptor(track_df):
    feats = np.array(track_df['feats'].values.tolist())
    conf = track_df['conf'].to_numpy()

    descriptor = (feats.T @ conf) / np.sum(conf)
    return descriptor

def compute_simultaneous(tracklets_df):

    # Filter alone tracks
    simultaneous = pd.DataFrame(tracklets_df.groupby('frameId')['trackIdx'].apply(list), columns=['trackIdx'])
    simultaneous = simultaneous[simultaneous['trackIdx'].apply(len) > 1]

    # Remove duplicated rows
    simultaneous['key'] = simultaneous['trackIdx'].apply(lambda x : str(sorted(x)))
    simultaneous.drop_duplicates(subset='key', inplace=True)

    # Generate pairs from simultaneous
    simultaneous['trackIdx'] = simultaneous['trackIdx'].apply(lambda x : list(combinations(x, 2)))
    simultaneous = simultaneous.explode('trackIdx')

    # Remove duplicated pairs
    simultaneous['key'] = simultaneous['trackIdx'].apply(lambda x : str(sorted(x)))
    simultaneous.drop_duplicates(subset='key', inplace=True)

    # Generate output
    simultaneous = pd.DataFrame(simultaneous['trackIdx'].tolist(), columns=['Idx1', 'Idx2'])
    idx1 = simultaneous['Idx1'].to_numpy()
    idx2 = simultaneous['Idx2'].to_numpy()

    return idx1, idx2

def compute_matrix(tracklets_df, descriptors_df, causality=True):
    descriptors_mat = descriptors_df.to_numpy().T

    dist_matrix = distance.cdist(descriptors_mat, descriptors_mat, metric='cosine') # 1 - cos(angle)

    if causality:
        # Apply anti-self-association and causality (a track can only be compared with a previous track)
        invalid_mask = np.full_like(dist_matrix, True, dtype=bool)
        invalid_mask = np.tril(invalid_mask)
    else:
        # We do not want to compare one track with itself (anti-self-association)
        invalid_mask = np.eye(dist_matrix.shape[0], dtype=bool)

    # 2 tracks that appers at the same time cannot be the same track
    idx1, idx2 = compute_simultaneous(tracklets_df)
    if len(idx1) > 0 and len(idx2) > 0:
        invalid_mask[idx1, idx2] = True
        invalid_mask[idx2, idx1] = True

    dist_matrix[invalid_mask] = np.inf

    return dist_matrix

def compute_merging(merging_matrix):
    valid = np.isfinite(merging_matrix.min(axis=0))
    paired = np.arange(len(valid))[valid]
    pair = merging_matrix.argmin(axis=0)[valid]

    merging_dict = dict()
    for ini, fin in zip(paired, pair):
        merging_dict[ini] = merging_dict.get(fin, fin)
    
    return merging_dict

def compare_first_with_last(gt_to_split, tracklets_df, gt_tracking_df):
    
    fr_dist = []
    px_dist = []
    app_dist = []
    err_app_dist = []
    
    for gt_id, split_ids in gt_to_split.items():
        if len(split_ids) > 1:

            splited = tracklets_df[tracklets_df['trackIdx'].apply(lambda x : x in split_ids)].copy().sort_values('frameId').reset_index()
            splited['groups'] = splited.index - splited['frameId']
            
            splited_fr = splited.groupby('groups')['frameId'].apply(list).reset_index().sort_values('groups', ascending=False).reset_index(drop=True)
            splited_fr['ini'] = splited_fr['frameId'].apply(lambda x : x[0])
            splited_fr['fin'] = splited_fr['frameId'].apply(lambda x : x[-1])
            
            for fin, ini in zip(splited_fr['fin'][:-1], splited_fr['ini'][1:]):
                fr_dist.append(ini - fin)

                ori = splited[splited['frameId'] == fin].iloc[0]
                dest = splited[splited['frameId'] == ini].iloc[0]

                pos_ori = np.array([ori['tlx'], ori['tly']])
                pos_dest = np.array([dest['tlx'], dest['tly']])
                px_dist.append( distance.euclidean(pos_ori, pos_dest) )

                app_ori = np.array(ori['feats'])
                app_dest = np.array(dest['feats'])
                app_dist.append(distance.cosine(app_ori, app_dest))

                for fr in range(int(fin), int(ini)):
                    noise = gt_tracking_df[ (gt_tracking_df['frameId'] == fr) & (gt_tracking_df['trackId'] != gt_id) ]
                    causal_noise = noise['feats'].apply(lambda x : distance.cosine(x, app_ori))
                    anticausal_noise = noise['feats'].apply(lambda x : distance.cosine(x, app_dest))

                    err_app_dist.extend(causal_noise.to_list())
                    err_app_dist.extend(anticausal_noise.to_list())
    
    return fr_dist, px_dist, app_dist, err_app_dist
