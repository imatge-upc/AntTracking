
from distutils.util import strtobool
from docopt import docopt


SUBSALG = 'MOG2'
VARTHRESHOLD = 15
FILTERFG = True
WRITEIMAGES = False
OUTPUTDIR = "out"
MINSIZE = 20
STARTWRITEFRAMES = 500
STOPFRAME = -1


DOCTEXT = f"""  
Use background substraction to detect ants in a video
  
Usage:
  ant_detection_pca.py <inputVideo> <detectionFile> [--subsAlg=<sa>] [--varThreshold=<vt>] [--filterFG=<ff>] [--writeImages=<wi>] [--outputDir=<od>] [--minSize=<ms>] [--startWriteFrames=<sw>] [--stopFrame=<sf>]
  ant_detection_pca.py -h | --help

Options:
  --subsAlg=<sa>            Background substraction algorithm (KNN/MOG2)  [default: {SUBSALG}]
  --varThreshold=<vt>       threshold [default: {VARTHRESHOLD}]
  --filterFG=<ff>           Whether to apply foreground filtering [default: {FILTERFG}]
  --writeImages=<wi>        Whether to save the fg images and boxes [default: {WRITEIMAGES}]
  --outputDir=<od>          Output folder where the images are stored [default: {OUTPUTDIR}]
  --minSize=<ms>            Minimum size (in pixels) of the width or height of the object to be considered [default: {MINSIZE}]
  --startWriteFrames=<sw>   Do not start writing frames until this threshold is reached [default: {STARTWRITEFRAMES}]
  --stopFrame=<sf>           Stop processing at this frame [default: {STOPFRAME}]
"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    input_video    = args['<inputVideo>']
    detection_file = args['<detectionFile>']

    subs_alg       = args['--subsAlg']
    var_thresh     = int(args['--varThreshold'])    
    filter_fg      = bool(strtobool(args['--filterFG']))
    write_images   = bool(strtobool(args['--writeImages']))
    out_dir        = args['--outputDir']
    min_size       = int(args['--minSize'])
    start_write    = int(args['--startWriteFrames'])
    stop_frame     = int(args['--stopFrame'])

    return input_video, detection_file, subs_alg, var_thresh, filter_fg, write_images, out_dir, min_size, start_write, stop_frame
