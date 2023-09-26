
from distutils.util import strtobool
from docopt import docopt


IMAGEWIDTH = 1920
IMAGEHEIGHT = 1080

TRACKTHRESHOLD = 0.6
IOUTHRESHOLD = 0.3

MINHITS = 3
USEBYTE = True
ASSOCIATIONFUNC = "iou"


DOCTEXT = f"""  
Track objects from detections in file (MOTChallenge format) using OC_SORT
  
Usage:
  ocsort_track.py <detFile> <trackFile> [--imageWidth=<iw>] [--imageHeight=<ih>] [--trackThreshold=<tt>] [--iouThreshold=<it>] [--minHits=<mh>] [--useByte=<ub>] [--associationFunc=<af>]
  ocsort_track.py -h | --help

Options:
  --imageWidth=<iw>         Image width  [default: {IMAGEWIDTH}]
  --imageHeight=<ih>        Image height [default: {IMAGEHEIGHT}]
  --trackThreshold=<tt>     Detection confidence threshold [default: {TRACKTHRESHOLD}]
  --iouThreshold=<it>       The iou threshold in Sort for matching [default: {IOUTHRESHOLD}]
  --minHits=<mh>            min hits to create track in SORT [default: {MINHITS}]
  --useByte=<ub>            use byte in tracking [default: {USEBYTE}]
  --associationFunc=<af>    Association function (iou/giou/ciou/diou/ct_dist) [default: {ASSOCIATIONFUNC}]
"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    detection_file = args['<detFile>']
    tracking_file  = args['<trackFile>']

    image_width    = int(args['--imageWidth'])
    image_height   = int(args['--imageHeight'])
    track_thresh   = float(args['--trackThreshold'])    
    iou_thresh     = float(args['--iouThreshold'])    
    min_hits       = int(args['--minHits'])    
    use_byte       = bool(strtobool(args['--useByte']))
    assoc_func     = args['--associationFunc']


    return detection_file, tracking_file, image_width, image_height, track_thresh, \
            iou_thresh, min_hits, use_byte, assoc_func
