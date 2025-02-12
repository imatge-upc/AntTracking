
import numpy as np


def propose_random_crop(trk, crop_width, crop_height, width, height):
    low = np.clip(np.array([trk[0] + trk[2] - crop_width, trk[1] + trk[3] - crop_height]), 0, None)
    high = np.array([trk[0] + 1, trk[1] + 1])

    initial = np.random.randint(low, high)
    final = initial + np.array([crop_width, crop_height], dtype=int)
    
    delta = np.clip(final - np.array([width, height]), 0, None)
    initial = (initial - delta).flatten().astype(int)
    final = (final - delta).flatten().astype(int)

    return initial, final, low, high

def find_intersections(tracks, initial, final):
    column_intersect = (tracks[:, 0] < final[0]) & ((tracks[:, 0] + tracks[:, 2]) >= initial[0])
    row_intersect = (tracks[:, 1] < final[1]) & ((tracks[:, 1] + tracks[:, 3]) >= initial[1])
    intersect = column_intersect & row_intersect

    return intersect

def desired_movments(unseen_tracks, initial, final):
    delta_w = np.clip(unseen_tracks[:, 0] - initial[0], None, 0) # max 0 (move left)
    delta_w = delta_w + np.clip(unseen_tracks[:, 0] + unseen_tracks[:, 2] - final[0], 0, None) # min 0 (move right)
    delta_h = np.clip(unseen_tracks[:, 1] - initial[1], None, 0)
    delta_h = delta_h + np.clip(unseen_tracks[:, 1] + unseen_tracks[:, 3] - final[1], 0, None)

    return delta_w.flatten().astype(int), delta_h.flatten().astype(int)

def mask_invalids(delta_w, delta_h, initial, final, low, high):
    delta_w2 = delta_w.copy()
    delta_h2 = delta_h.copy()

    max_left = low[0] - initial[0]
    max_right = high[0] - final[0]
    max_up = low[1] - initial[1]
    max_down = high[1] - final[1]

    invalid = (delta_w < max_left) | (delta_w > max_right) | (delta_h < max_up) | (delta_h > max_down)

    delta_w2[invalid] = 0
    delta_h2[invalid] = 0

    return delta_w2.flatten().astype(int), delta_h2.flatten().astype(int)

def undesired_movments(unseen_tracks, initial, final):
    no_delta_left = np.clip(unseen_tracks[:, 0] + unseen_tracks[:, 2] - final[0], None, 0)
    no_delta_right = np.clip(unseen_tracks[:, 0] - initial[0], 0, None)
    no_delta_up = np.clip(unseen_tracks[:, 1] + unseen_tracks[:, 3] - final[1], None, 0)
    no_delta_down = np.clip(unseen_tracks[:, 1] - initial[1], 0, None)

    return no_delta_left, no_delta_right, no_delta_up, no_delta_down

def best_movments(delta_w, delta_h, no_delta_left, no_delta_right, no_delta_up, no_delta_down):

    delta_w = delta_w.reshape((1, -1)) # 1, N
    delta_h = delta_h.reshape((1, -1)) # 1, N

    no_delta_left = no_delta_left.reshape((-1, 1)) # M, 1
    no_delta_right = no_delta_right.reshape((-1, 1)) # M, 1
    no_delta_up = no_delta_up.reshape((-1, 1)) # M, 1
    no_delta_down = no_delta_down.reshape((-1, 1)) # M, 1

    left_scores = ((delta_w >= no_delta_left) & (delta_w <= 0)).sum(axis=0) # N
    right_scores = ((delta_w <= no_delta_right) & (delta_w >= 0)).sum(axis=0) # N
    down_scores = ((delta_h >= no_delta_up) & (delta_h <= 0)).sum(axis=0) # N
    up_scores = ((delta_h <= no_delta_down) & (delta_h >= 0)).sum(axis=0) # N

    left_up_score = np.min([left_scores, up_scores], axis=0) # N
    right_up_score = np.min([right_scores, up_scores], axis=0) # N
    left_down_score = np.min([left_scores, down_scores], axis=0) # N
    right_down_score = np.min([right_scores, down_scores], axis=0) # N

    scores = np.array([left_up_score, right_up_score, left_down_score, right_down_score]) # 4, N
    _, idx_delta_w_h = np.unravel_index(np.argmax(scores).flatten()[0], scores.shape)

    delta = np.array([delta_w[0, idx_delta_w_h], delta_h[0, idx_delta_w_h]]).flatten().astype(int)

    return delta

def random_delta_shake(unseen_tracks, initial, final, delta, width, height):
    initial2 = initial + delta
    final2 = final + delta

    no_delta_left, no_delta_right, no_delta_up, no_delta_down = undesired_movments(unseen_tracks, initial2, final2)
    
    min_w = np.maximum(np.max(no_delta_left), -initial2[0])
    max_w = np.minimum(np.min(no_delta_right), width - final[0])
    min_h = np.maximum(np.max(no_delta_up), -initial2[1])
    max_h = np.minimum(np.min(no_delta_down), height - final[1])

    delta = delta + np.random.randint([min_w, min_h], [max_w + 1, max_h + 1]).flatten()
    return delta

def improve_crop(unseen_tracks, initial, final, low, high, width, height):
    delta_w1, delta_h1 = desired_movments(unseen_tracks, initial, final)
    delta_w2, delta_h2 = mask_invalids(delta_w1, delta_h1, initial, final, low, high)
    
    no_delta_left, no_delta_right, no_delta_up, no_delta_down = undesired_movments(unseen_tracks, initial, final)

    delta_0 = best_movments(delta_w2, delta_h2, no_delta_left, no_delta_right, no_delta_up, no_delta_down)

    delta = random_delta_shake(unseen_tracks, initial, final, delta_0, width, height)
    
    initial = (initial + delta).flatten().astype(int)
    final = (final + delta).flatten().astype(int)

    return initial, final
