
from docopt import docopt


CONFIG_FILE = 'runs/apparence/train01_colonia_256_128/config.yaml'
WEIGHTS_PATH = 'runs/apparence/train01_colonia_256_128/model_best.pth'

DOCTEXT = f""" 
Usage:
  appearance_tracks.py <input_video> <detFile> <output_file> [--config=<cf>] [--weights=<wp>]
  appearance_tracks.py -h | --help

Options:
  --config=<cf>       Config file from the fastreid model. [default: {CONFIG_FILE}]
  --weights=<wp>      Weights path from the fastreid model. [default: {WEIGHTS_PATH}]

"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    input_video = args['<input_video>']
    detection_file = args['<detFile>']
    tracking_file  = args['<output_file>']

    config_file = args['--config']
    weights_path = args['--weights']

    return input_video, detection_file, tracking_file, config_file, weights_path
