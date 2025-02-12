
from collections import defaultdict
from docopt import docopt
import pathlib
import sys

from ceab_ants.colors.classifiers.color_gmm import ColorGMM
from ceab_ants.io.crop_loader import ColorCropLoader


def load_models(gmms_path):
    models = defaultdict()
    for filename in gmms_path:
        model = ColorGMM()
        model.load(filename)
        models[model.color] = model
    return models

def score_img(img, models_dict):
    scores = { color : model.score(img.reshape(-1, 3))  for color, model in models_dict.items() }
    return scores

def classify_img(img, models_dict):
    scores = [( color, model.score(img.reshape(-1, 3)) ) for color, model in models_dict.items()]
    color = max(scores, key=lambda x : x[1])[0]
    return color
    
def main(gmms_path, dataset_path, data_path, output):
    dataset = ColorCropLoader(dataset_path, data_path, labels=False)
    models = load_models(gmms_path)

    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        [print(f"{path},{classify_img(img, models)}", end='\n', file=f) for img, path in dataset]

def main_scores(gmms_path, dataset_path, data_path, output, colors):
    dataset = ColorCropLoader(dataset_path, data_path, labels=False)
    models = load_models(gmms_path)

    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        print(f"path,{','.join(colors)}", end='\n', file=f)
        for img, path in dataset:
            scores = score_img(img, models)
            print(f"{path},{','.join([str(scores.get(color, '')) for color in colors])}", end='\n', file=f)


DOCTEXT = """
    Usage:
      classify_colors.py <gmms>... --dataset_path=<d> --data_path=<dp> --output=<o>
      classify_colors.py scores <gmms>... --dataset_path=<d> --data_path=<dp> --output=<o> --colors=<c>...
      classify_colors.py -h | --help
"""


if __name__ == "__main__":
    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    gmms_path = args["<gmms>"]
    dataset_path = args["--dataset_path"]
    data_path = args["--data_path"]
    output = args["--output"]

    scores = args["scores"]
    colors = args["--colors"]
    
    if scores:
        main_scores(gmms_path, dataset_path, data_path, output, colors)
    else:
        main(gmms_path, dataset_path, data_path, output)
