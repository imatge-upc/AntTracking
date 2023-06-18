
from docopt import docopt


DOCTEXT = f""" 
Usage:
  apparence_tracks.py <input_video> <detFile> <trackFile>
  apparence_tracks.py -h | --help

"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    input_video = args['<input_video>']
    detection_file = args['<detFile>']
    tracking_file  = args['<trackFile>']


    return input_video, detection_file, tracking_file
