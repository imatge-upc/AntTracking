
from collections import defaultdict
import numpy as np
import pathlib
import sys

from ceab_ants.colors.classifiers.color_gmm import ColorGMM
from ceab_ants.io.crop_loader import ColorCropLoader

from docopts.help_train_colors import parse_args


SK1_VERBOSE = 1 << 0
SK2_VERBOSE = 1 << 1
MEAN_VERBOSE = 1 << 2
SAVE_VERBOSE = 1 << 3
LAST_VERBOSE = 1 << 4


def load_all(dataset_path, data_path, labels=True):
    dataset = ColorCropLoader(dataset_path, data_path, labels=labels)
    loaded_dataset = defaultdict(list)
    for img, color in dataset: # If labels=False then color is img subpath
        loaded_dataset[color].append(img.reshape(-1, 3))
    return loaded_dataset

def main(dataset_path, data_path, basename, num_gauss, cov_type, num_iters, num_init, verbose=False, sk_verbose_interval=200):
    loaded_dataset = load_all(dataset_path, data_path, labels=True)

    pathlib.Path(basename).parent.mkdir(parents=True, exist_ok=True)

    # Pixel color distribution
    models = dict()
    for color, data in loaded_dataset.items():
        models[color] = ColorGMM(
            color,
            num_gauss, 
            cov_type, 
            num_iters, 
            num_init, 
            verbose=(verbose & SK1_VERBOSE) | (verbose & SK2_VERBOSE), 
            verbose_interval=sk_verbose_interval
        )
        models[color].train(np.concatenate(data))

        if (verbose & MEAN_VERBOSE) : print(f"{color} means: \n", models[color].means_)

        models[color].save(basename)
        
        if (verbose & SAVE_VERBOSE) : print(f"{basename}_{color}.dill") # TODO: Maybe add timestamp on SAVE and MEAN verbose print
    
    if (verbose & LAST_VERBOSE):
        print("\n\n---- SUMMARY ----\n")
        for _, color in models:
            print(f"{basename}_{color}.dill")


if __name__ == "__main__":

    (dataset_path, 
     data_path, 
     basename, 
     num_gauss, 
     cov_type, 
     num_iters, 
     num_init, 
     verbose, 
     sk_verbose_interval) = parse_args(sys.argv)

    main(
        dataset_path, 
        data_path, 
        basename, 
        num_gauss, 
        cov_type, 
        num_iters, 
        num_init,
        verbose=verbose, 
        sk_verbose_interval=sk_verbose_interval
    )
