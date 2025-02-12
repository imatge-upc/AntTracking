
from distutils.util import strtobool
from docopt import docopt


DOCTEXT = """
Reads a tracking file in the MOTchallenge format, reads the tracked video and plots the rectangles over the corresponding objects, then saves the video.

Usage:
  plot_tracks.py <trackFile> <inputVideo> <outputVideo>  [--downsampleVideo=<dv>] [--startFrame=<sf>] [--maxFrame=<mf>]
  plot_tracks.py -h | --help

Options:
  --downsampleVideo=<dv>         Downsample the output video by a factor of 2 [default: False]
  --startFrame=<sf>              First frame to process [default: 1]
  --maxFrame=<mf>                Stop processing at this frame [default: -1]
"""


def parse_args(argv):
    
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    tracking_file    = args['<trackFile>']
    input_video      = args['<inputVideo>']
    out_video        = args['<outputVideo>']
    downsample_video = bool(strtobool(args['--downsampleVideo']))
    start_frame      = int(args['--startFrame'])
    max_frame        = int(args['--maxFrame'])

    return tracking_file, input_video, out_video, downsample_video, start_frame, max_frame
