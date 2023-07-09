
from distutils.util import strtobool
from docopt import docopt


DOCTEXT = f"""  
Plot the GT and predictes tracks for a given sequence. GT and track files should be in MOT20 format:
Tracking and annotation files are simple comma-separated value (CSV) files. Each line represents one 
object instance and contains 9 values for GT and 10 for results files. The first number indicates in 
which frame the object appears, while the second number identifies that object as belonging to a 
trajectory by assigning a unique ID. Each object can be assigned to only one trajectory. The next 
four numbers indicate the position of the bounding box in 2D image coordinates. The position is 
indicated by the top-left corner as well as width and height of the bounding box. 

For the ground truth and results files, the 7th value acts as a flag whether the entry is to be 
considered. A value of 0 means that this particular instance is ignored in the evaluation, while a 
value of 1 is used to mark it as active. In the GT files, the 8th number indicates the type of object 
annotated. A value of 1 should be used (scoring and visualization are class-agnostic). The last number
shows the visibility ratio of each bounding box. 

In the results files, the 7th, 8th, 9th and 10th numbers are -1.

GT file example: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <valid>, <1>, <ignored>, 
Results file example: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <ignored>, <ignored>, <ignored>, <ignored>

Source: MOT20: A benchmark for multi object tracking in crowded scenes. https://arxiv.org/pdf/2003.09003.pdf

Usage:
  plot_tracks.py <gtFolder> <tracksFolder> [--trackerList=<tl>] [--oneFile=<of>]
  plot_tracks.py -h | --help

  <gtFolder>              Folder with the gt annotations (MOT20 format). For instance: data/gt/mot_challenge/
  <trackFolder>           Folder with the tracking results (MOT20 format). For instance: data/trackers/mot_challenge/
  -------------
  USAGE EXAMPLE: cd /imatge/morros/workspace/mediapro/post_tracking_reid;  python plot_tracks.py data/gt/mot_challenge/ data/trackers/mot_challenge/ --trackerList=OTrack
Options:
  --trackerList=<tl>      Name of the trackers to evaluate. String separated by commas [default: '']
  --oneFile=<of>          Plot just <gtFolder> [default: False]


plot_tracks_data/
├── gt
│   └── mot_challenge
│       ├── MOT20-train
│       │   └── results
│       │       ├── gt
│       │       │   └── gt.txt
│       │       └── seqinfo.ini
│       └── seqmaps
│           └── MOT20-train.txt
└── trackers
    └── mot_challenge
        └── MOT20-train
            └── OTrack
                └── data
                    └── results.txt
"""


def parse_args(argv):
    args = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    gt_folder     = args["<gtFolder>"]          # /mnt/gpid08/datasets/sports_analytics/SoccerNet/tracking/train/SNMOT-170/img1
    tracks_folder = args["<tracksFolder>"]     # /mnt/gpid08/datasets/sports_analytics/SoccerNet/tracking/train/SNMOT-170/gt/gt.txt
    tracker_list  = args['--trackerList'].split(',')
    one_file      = bool(strtobool(args['--oneFile']))


    return gt_folder, tracks_folder, tracker_list, one_file
