
from docopt import docopt
import numpy as np
import sys


class PrecomputedMOTTracker():

    def __init__(self, seq_path=None, first_frame=1):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')
        self.first_frame = first_frame
        
        self.last_frame = int(self.seq_dets[:, 0].max())

        self.current_frame = first_frame
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def __call__(self, frame):

        tracks = self.seq_dets[self.seq_dets[:, 0] == self.current_frame]
        self.current_frame += 1

        return tracks.reshape(-1, 10)


DOCTEXT = f"""
Usage:
  minimum_id.py <seq_path> <output_file>
"""


if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    seq_path = args['<seq_path>']
    output_file = args['<output_file>']

    tracker = PrecomputedMOTTracker(seq_path)

    id_transform = dict()

    results = []
    for frame in range(1, tracker.last_frame + 1):

        tracks = tracker(frame)

        for trk in tracks:
            if trk[1] not in id_transform.keys():
                id_transform[trk[1]] = len(id_transform) + 1
            results.append(f"{int(trk[0])},{id_transform[trk[1]]},{trk[2]:.2f},{trk[3]:.2f},{trk[4]:.2f},{trk[5]:.2f},{trk[6]:.1f},{int(trk[7])},{int(trk[8])},{int(trk[9])}\n")
    
    with open(output_file, 'w') as f:
        f.writelines(results)
