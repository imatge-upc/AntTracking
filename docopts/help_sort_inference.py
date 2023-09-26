
from docopt import docopt


SEQ_PATH = 'data'
MAX_AGE = 1
MIN_HITS = 3
IOU_THRESHOLD = 0.3


DOCTEXT = f"""  
SORT demo
  
Usage:
  sort_inference.py <seq_path> [--max_age=<ma>] [--min_hits=<mh>] [--iou_threshold=<it>]
  sort_inference.py -h | --help

Options:
  --max_age=<ma>            Maximum number of frames to keep alive a track without associated detections. [default: {MAX_AGE}]
  --min_hits=<mh>           Minimum number of associated detections before track is initialised. [default: {MIN_HITS}]
  --iou_threshold=<it>      Minimum IOU for match. [default: {IOU_THRESHOLD}]
"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    seq_path = args['<seq_path>']
    max_age = int(args['--max_age'])
    min_hits = int(args['--min_hits'])
    iou_threshold = float(args['--iou_threshold'])

    return seq_path, max_age, min_hits, iou_threshold
