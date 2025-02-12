
from collections import defaultdict
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np


def prepare_canvas(bins_per_color=256, colors=3, ticks_per_color=4, colormap=None):
    nbins = bins_per_color * colors
    range_ = [0, nbins]
    if colormap is None and colors == 3 : colormap = ["red", "green", "blue"]
    elif colormap is None : colormap = cm.Set1(range(colors))

    fig = plt.figure()
    ax = fig.gca()

    img_hist, hist_bins = np.histogram([], bins=nbins, range=range_)

    hist_bins_ticks = hist_bins[:-1].copy().astype(int)
    for i in range(colors):
        hist_bins_ticks[bins_per_color * i : bins_per_color * (i + 1)] -= 256 * i

    hist_bins = (hist_bins[:-1] + hist_bins[1:]) / 2

    ax.bar(hist_bins, img_hist)

    # apply 1 color to all ticks, then the next color to all ticks
    i_ticks = [i + bins_per_color * c for c in range(colors) for i in range(0, bins_per_color, bins_per_color // ticks_per_color)]
    ax.set_xticks([hist_bins[i] for i in i_ticks], [hist_bins_ticks[i] for i in i_ticks])
    for c in range(colors):
        for i in range(ticks_per_color):
            ax.get_xticklabels()[i + ticks_per_color * c].set_color(colormap[c])

    return fig, ax, nbins, range_, hist_bins

def image_hist(img, nbins, range_, norm=False):
    # TODO: fer 3 histogrames i concatenar, modificar el codi on toqui
    assert 256 % (nbins / img.shape[-1]) == 0 # Si nbins no es divisor, els maxims d'un canal i el següent es junten 
    img = img.copy()

    for i in range(img.shape[-1]):
        img[..., i] += 256 * i

    img = img.flatten()

    img_hist, _ = np.histogram(img, bins=nbins, range=range_)
    if norm : img_hist = img_hist.astype(float) / len(img)

    return img_hist

def hist_1d_to_3d(color_hist, bins_per_color=256, ncolors=3):
    return np.stack([color_hist[bins_per_color * i : bins_per_color * (i + 1)] for i in range(ncolors)], axis=-1)

def dataset_hist(dataset, nbins, range_, norm=False, flat=True):
    histograms = defaultdict(list)
    for img, color in dataset:
        img_hist = image_hist(img, nbins, range_, norm=norm)
        if not flat : img_hist = hist_1d_to_3d(img_hist, nbins // img.shape[-1], img.shape[-1])
        histograms[color].append(img_hist)
    return histograms
