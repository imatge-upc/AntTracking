
from docopt import docopt


VAR_THRESH = 15
MIN_SIZE = 20
START_WRITE = 0
NUM_FRAMES = -1
LR_TRAIN = -1
LR = -1
FRAMES_TRAIN = 100000
QUEUE_SIZE = 4

DOCTEXT = f"""
Usage:
  ant_detection_bgfg_oriented.py <input_video> <detection_file> [--var_thresh=<vt>] [--min_size=<ms>] [--start_write=<sw>] [--num_frames=<nf>] [--lr_train=<lrt>] [--lr=<lr>] [--frames_train=<ft>] [--queue_size=<qs>]
  ant_detection_bgfg_oriented.py -h | --help

Options:
  --var_thresh=<vt>       Variance threshold between background and current frame foreground [default: {VAR_THRESH}]
  --min_size=<ms>         Minimum size of a detection to be considered a valid detection [default: {MIN_SIZE}]
  --start_write=<sw>      First frame to include on the detection file, previous frames are used for training [default: {START_WRITE}]
  --num_frames=<nf>       Number of frames to process after start_write. A value of 0 or negative means process until the end of the video [default: {NUM_FRAMES}]
  --lr_train=<lrt>        Learning rate for training (negative for automatic) [default: {LR_TRAIN}]
  --lr=<lr>               Learning rate for applying the model (negative for automatic) [default: {LR}]
  --frames_train=<ft>     Number of frames used for training before applying the model, set at start_write [default: {FRAMES_TRAIN}]
  --queue_size=<qs>       Maximum size of the frame queue [default: {QUEUE_SIZE}]

"""

def parse_args(argv):
    
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    input_video = args['<input_video>']
    detection_file = args['<detection_file>']
    var_thresh = float(args['--var_thresh'])
    min_size = int(args['--min_size'])
    start_write = int(args['--start_write'])
    num_frames = int(args['--num_frames'])
    lr_train = float(args['--lr_train'])
    lr = float(args['--lr'])
    frames_train = int(args['--frames_train'])
    queue_size = int(args['--queue_size'])

    return input_video, detection_file, var_thresh, min_size, start_write, num_frames, lr_train, lr, frames_train, queue_size
