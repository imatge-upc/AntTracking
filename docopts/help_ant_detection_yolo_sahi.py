
from distutils.util import strtobool
from docopt import docopt


IMGSZ = 320
STOPFRAME = -1


DOCTEXT = f"""  
Use background substraction to detect ants in a video. ('runs/detect/train2/weights/best.pt')
  
Usage:
  ant_detection_yolo_sahi.py <inputVideo> <detectionFile> <weights_path> [--imgsz=<is>] [--stopFrame=<sf>]
  ant_detection_yolo_sahi.py -h | --help

Options:
  --imgsz=<is>               Image size  [default: {IMGSZ}]
  --stopFrame=<sf>           Stop processing at this frame [default: {STOPFRAME}]
"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    input_video    = args['<inputVideo>']
    detection_file = args['<detectionFile>']
    weights_path   = args['<weights_path>']

    imgsz       = int(args['--imgsz'])
    stop_frame  = int(args['--stopFrame'])

    return input_video, detection_file, weights_path, imgsz, stop_frame
