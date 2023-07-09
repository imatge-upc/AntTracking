
from distutils.util import strtobool
from docopt import docopt


IMGSZ = 320
STOPFRAME = -1
CONF = 0.3


DOCTEXT = f"""  
Use background substraction to detect ants in a video. ('runs/detect/train2/weights/best.pt')
  
Usage:
  ant_detection_yolo_sahi_v2.py <inputVideo> <detectionFile> <weights_path> [--imgsz=<is>] [--stopFrame=<sf>] [--conf=<c>]
  ant_detection_yolo_sahi_v2.py -h | --help

Options:
  --imgsz=<is>               Image size  [default: {IMGSZ}]
  --stopFrame=<sf>           Stop processing at this frame [default: {STOPFRAME}]
  --conf=<c>                 Confidence [default: {CONF}].
"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    input_video    = args['<inputVideo>']
    detection_file = args['<detectionFile>']
    weights_path   = args['<weights_path>']

    imgsz       = int(args['--imgsz'])
    stop_frame  = int(args['--stopFrame'])
    conf        = float(args['--conf'])

    return input_video, detection_file, weights_path, imgsz, stop_frame, conf
