
from distutils.util import strtobool
from docopt import docopt


IOUTHRESHOLD = 0.3
MINHITS = 3
MAX_AGE = 1


DOCTEXT = f"""  
Track objects from detections in file (MOTChallenge format) using Deep_SORT
  
Usage:
  deepsort_track.py <detFile> <trackFile> [--iouThreshold=<it>] [--minHits=<mh>] [--max_age=<ma>]
  deepsort_track.py -h | --help

Options:
  --iouThreshold=<it>       The iou threshold in Sort for matching [default: {IOUTHRESHOLD}]
  --max_age=<ma>            Maximum number of frames to keep alive a track without associated detections. [default: {MAX_AGE}]
  --minHits=<mh>            min hits to create track in SORT [default: {MINHITS}]
"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    detection_file = args['<detFile>']
    tracking_file  = args['<trackFile>']

    iou_thresh     = float(args['--iouThreshold'])    
    max_age        = int(args['--max_age'])    
    min_hits       = int(args['--minHits'])    

    return detection_file, tracking_file, max_age, min_hits, iou_thresh 
