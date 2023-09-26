
from matplotlib import pyplot as plt
import numpy as np


def normalized_hist(data, max_freq=1, n_ticks=5):
    plt.ylim(0, len(data) * max_freq)
    plt.gca().set_yticks(np.linspace(0, len(data) * max_freq, n_ticks), labels=np.around(np.linspace(0, max_freq, n_ticks), decimals=2))
    plt.ylabel(f'Normalized frequency')


def plot_global_feats(descriptors):

    desc = np.vstack(list(descriptors.values()))
    scale = np.sum(np.abs(desc), axis=0) / desc.shape[0]
    range_ = np.sort(np.max(desc, axis=0) - np.min(desc, axis=0))[::-1]

    fig1 = plt.figure()
    fig1.suptitle('Histogram of Global Features')
    ax = fig1.add_subplot()
    ax.set_title('hist( np.sum(np.abs(desc), axis=0) / desc.shape[0], log=True)')
    ax.hist(scale, log=True)

    fig2 = plt.figure()
    fig2.suptitle('Dynamic Range of Features (sorted and normalized)')
    ax = fig2.add_subplot()
    ax.set_title('np.max(desc, axis=0) - np.min(desc, axis=0)')
    ax.bar(np.arange(desc.shape[1]), range_ / np.max(range_))

    return fig1, fig2

def plot_distance_info(dist_matrix, gt_to_split):

    correct_dist = []
    incorrect_dist = []

    valid = np.isfinite(dist_matrix)
    for same_idxs in gt_to_split.values():
        diff_idxs = np.arange(len(dist_matrix))[~np.in1d(np.arange(len(dist_matrix)), same_idxs)]

        correct_mask = np.ix_(same_idxs, same_idxs)
        incorrect_mask = np.ix_(diff_idxs, diff_idxs)

        correct_dist.extend(dist_matrix[correct_mask].flatten()[valid[correct_mask].flatten()].tolist())
        incorrect_dist.extend(dist_matrix[incorrect_mask].flatten()[valid[incorrect_mask].flatten()].tolist())

    fig1 = plt.figure()
    fig1.suptitle('Tracklets Joinability\n(Ideal Case with Tracklets without False Positives)')
    ax = fig1.add_subplot()
    ax.hist([correct_dist, incorrect_dist], bins=100, log=False, label=['same track', 'different track'], density=True)
    #ax.hist([correct_dist, incorrect_dist], bins=np.logspace(-5, 0, 100), log=False, label=['same track', 'different track'], density=True)
    #ax.set_xscale('log')
    ax.set_xlabel('Cosine distance')
    ax.set_ylabel('Probability Density of Tracklet Distances')
    ax.legend()

    return fig1

def plot_first_last(fr_dist, px_dist, app_dist, err_app_dist):

    fig1 = plt.figure(figsize=(6.4 * 2.5, 4.8 * 1.25))
    fig1.suptitle(f'Space-temporal scope of a meeting of ants')

    ax = fig1.add_subplot(1, 2, 1)
    ax.set_title(f'Histogram of duration in frames')
    ax.hist(fr_dist, range=[0, 50], bins=25)
    normalized_hist(fr_dist, max_freq=0.6, n_ticks=7)
    plt.xticks(np.arange(0, 50, 2) + 1, np.arange(1, 50, 2))
    ax.set_xlabel('Time (fr)')
    
    ax = fig1.add_subplot(1, 2, 2)
    ax.set_title(f'Histogram of displacement in pixels')
    ax.hist(px_dist, range=[0, 50], bins=25)
    normalized_hist(px_dist, max_freq=0.3, n_ticks=7)
    plt.xticks(np.arange(0, 50, 2) + 1, np.arange(1, 50, 2))
    ax.set_xlabel('Distance (px)')

    fig2 = plt.figure()
    fig2.suptitle('Joinability of Individual Crops near a Splitting Point')
    ax = fig2.add_subplot()
    """
    ax.hist([app_dist, err_app_dist], bins=np.logspace(-5, 0, 100), log=True, label=['last to first', 'last/first to error'])
    ax.set_xscale('log')
    ax.set_xlabel('Apparence vectors cosine distance')
    """
    ax.hist([app_dist, err_app_dist], bins=100, log=False, label=['last to first', 'last/first to error'], density=True)
    ax.set_xlabel('Cosine distance')
    ax.set_ylabel('Probability Density of Tracklet Distances')
    ax.legend()

    return fig1, fig2
