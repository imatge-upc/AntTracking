
from distutils.util import strtobool
from docopt import docopt


IMGSZ = 320
STOPFRAME = -1
OVERLAP = 0.2
CONF = 0.25
INITIALFRAME = 0


DOCTEXT = f"""    
Usage:
  ant_detection_yolo_bigNMS.py <inputVideo> <detectionFile> <weights_path> [--imgsz=<is>] [--overlap=<o>] [--conf=<c>] [--stopFrame=<sf>] [--initialFrame=<if>]
  ant_detection_yolo_bigNMS.py -h | --help

Options:
  --imgsz=<is>               Image size  [default: {IMGSZ}]
  --overlap=<o>              Windows overlap [default: {OVERLAP}].
  --conf=<c>                 Confidence [default: {CONF}].
  --stopFrame=<sf>           Stop processing at this frame [default: {STOPFRAME}]
  --initialFrame=<if>        Init processing at this frame (from 0 to #frames-1) [default: {INITIALFRAME}]
"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    input_video    = args['<inputVideo>']
    detection_file = args['<detectionFile>']
    weights_path   = args['<weights_path>']

    imgsz         = int(args['--imgsz'])
    overlap       = float(args['--overlap'])
    conf          = float(args['--conf'])
    stop_frame    = int(args['--stopFrame'])
    initial_frame = max(int(args['--initialFrame']), 0)

    return input_video, detection_file, weights_path, imgsz, overlap, conf, stop_frame, initial_frame
