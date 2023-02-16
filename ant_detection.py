"""  
Use background substraction to detect ants in a video
  
Usage:
  ant_detection.py <inputVideo> <detectionFile> [--subsAlg=<sa>] [--varThreshold=<vt>] [--filterFG=<ff>] [--writeImages=<wi>] [--outputDir=<od>] [--minSize=<ms>] [--startWriteFrames=<sw>] [--stopFrame=<sf>]
  ant_detection.py -h | --help

Options:
  --subsAlg=<sa>            Background substraction algorithm (KNN/MOG2)  [default: MOG2]
  --varThreshold=<vt>       threshold [default: 15]
  --filterFG=<ff>           Whether to apply foreground filtering [default: True]
  --writeImages=<wi>        Whether to save the fg images and boxes [default: False]
  --outputDir=<od>          Output folder where the images are stored [default: out]
  --minSize=<ms>            Minimum size (in pixels) of the width or height of the object to be considered [default: 20]
  --startWriteFrames=<sw>   Do not start writing frames until this threshold is reached [default: 500]
  --endFrame<ef>            Stop processing at this frame [default: -1]
"""

import cv2 as cv
import argparse
import numpy as np
from docopt import docopt
from distutils.util import strtobool
import sys


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
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


    if subs_alg == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2(varThreshold=var_thresh, detectShadows=False)
    else:
        backSub = cv.createBackgroundSubtractorKNN()

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(input_video))
    if not capture.isOpened():
        print('Unable to open: ' + input_video, file=sys.stderr)
        exit(0)

    first = True
    se1 = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    se2 = cv.getStructuringElement(cv.MORPH_RECT, (2,2))

    fr = 1

    with open(detection_file, 'w') as out_file:
        while True:
            if stop_frame > 0 and fr > stop_frame:
                break
            
            if fr%500 == 0:
                print (f'Processing frame {fr}', file=sys.stderr)
            ret, frame = capture.read()
            if frame is None:
                break

            h,w,d = frame.shape
            if first == True:
                out_ima = np.zeros((h,2*w,d), dtype=np.uint8)
                first = False
                
            fgMask = backSub.apply(frame)

            if fr < 500:
                fr = fr + 1
                continue
    
            # Remove noise  https://stackoverflow.com/questions/30369031/remove-spurious-small-islands-of-noise-in-an-image-python-opencv
            if filter_fg:
                fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, se1)
                fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN,  se2)
                
            out_ima[:,0:w,:] = frame
            out_ima[:,w:, 0] = fgMask
            out_ima[:,w:, 1] = fgMask
            out_ima[:,w:, 2] = fgMask


            analysis = cv.connectedComponentsWithStats(fgMask, 4, cv.CV_32S)
            (totalLabels, labels, stats, centroids) = analysis


            for ii in range(1, totalLabels):
                # extract the connected component statistics for the current
                # label
                x    = stats[ii, cv.CC_STAT_LEFT]
                y    = stats[ii, cv.CC_STAT_TOP]
                w    = stats[ii, cv.CC_STAT_WIDTH]
                h    = stats[ii, cv.CC_STAT_HEIGHT]
                #area = stats[ii, cv.CC_STAT_AREA]
                if w > min_size and h > min_size:
                    #print (f'Warning: small object: x={x}, y={y}, w={w}, h={h}', file=sys.stderr)
                    print (f'{fr}, -1, {x}, {y}, {w}, {h}, 1, -1, -1, -1', file=out_file)

            if write_images and fr > start_write:
                cv.imwrite (f'{out_dir}/out_{fr:06d}.png', out_ima)
        
            fr = fr + 1

    # Release the video capture and writer objects
    capture.release()
