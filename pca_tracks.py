
from contextlib import contextmanager
import cv2 as cv
import numpy as np
import os
import sys
import time

from docopts.help_pca_tracks import parse_args


np.random.seed(0)


@contextmanager
def VideoCapture(input_video):

    # findFileOrKeep allows more searching paths
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(input_video))
    if not capture.isOpened():
        print('Unable to open: ' + input_video, file=sys.stderr)
        exit(0)

    try:
        yield capture
    finally:
        # Release the video capture object at the end
        capture.release()


def crop_pca_orientation(frame, bbox, background_th, min_size=20):
    crop = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    pts = np.argwhere(crop < background_th).reshape(-1, 2).astype(np.float32)
    if len(pts) < min_size : return 0

    mean = np.empty((0))
    _, eigenvectors, _ = cv.PCACompute2(pts, mean)

    angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
    return angle

class PrecomputedMOTTrackerPCA():

    def __init__(self, seq_path=None, first_frame=1, verbose=False):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')
        self.last_frame = int(self.seq_dets[:, 0].max())

        self.first_frame = first_frame
        self.current_frame = first_frame

        self.verbose = verbose
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def __call__(self, frame):

        if self.verbose and (self.current_frame % 500 == 0):
            print (f'Processing frame {self.current_frame}', file=sys.stderr)

        tcks = self.seq_dets[self.seq_dets[:, 0] == self.current_frame, :]
        self.current_frame += 1
        
        gray_frame = len(tcks) == 0 or cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        background_th = len(tcks) == 0 or np.mean(gray_frame) * 0.5

        angles = np.asarray([crop_pca_orientation(gray_frame, tck[2:6].astype(int), background_th) for tck in tcks])
        if len(tcks) > 0:
            tcks[:, -1] = angles

        return tcks


if __name__ == '__main__':

    input_video, detection_file, tracking_file = parse_args(sys.argv)

    total_time = 0.0
    total_frames = 0

    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    
    model = PrecomputedMOTTrackerPCA(detection_file, verbose=True)

    print(f'Processing {detection_file}')

    # Apply the model
    with VideoCapture(input_video) as capture:
        # As we do not want to load the video, this trick is enough
        results = []
        for frame_id in range(1, model.last_frame + 1):
            total_frames += 1

            _, frame = capture.read()
            if frame is None:
                print (f'Frame {frame_id} is None', file=sys.stderr)
                break

            start_time = time.time()
            online_targets = model(frame)
            cycle_time = time.time() - start_time

            total_time += cycle_time

            for t in online_targets:
                results.append(f"{t[0]},{t[1]},{t[2]:.2f},{t[3]:.2f},{t[4]:.2f},{t[5]:.2f},1.0,-1,-1,{t[-1]:.2f}\n")

    with open(os.path.join('OUTPUT', tracking_file), 'w') as f:
        f.writelines(results)

    print(f"Total Tracking took: {total_time:.3f} seconds for {total_frames:d} frames or {total_frames / total_time:.1f} FPS")
