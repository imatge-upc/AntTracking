
# NOTE: I don't remember if there is a difference. I prefre the wp2_visualization codes (although I'm still finishing them). wp1_evaluation should have something like HOTA metrics

import cv2
import numpy as np
import pandas as pd
import sys

from ceab_ants.io.video_contextmanager import VideoCapture, VideoWritter

from docopts.help_plot_tracks import parse_args
from utils.colours import colours


def trackinfo (df):
    # list of unique tracks
    tracks = pd.unique(df['trackId'])

    # Create an empty table that will contain, for each track, the list of frames and the list of bboxes
    dftrack = pd.DataFrame(columns = ['trackId', 'frames', 'centers'])

    # For each track ...
    for ii, track in enumerate(tracks):
        # Create a new dataframe with all the frames that contain this track
        dft = df.loc[df['trackId'] == track]

        frames = dft['frameId'].tolist()
        # List of bboxes of this track
        centers = (dft[['tlx', 'tly']].to_numpy() + dft[['width', 'height']].to_numpy() // 2).round().astype(int).tolist()
        
        # Add a new row to the table
        dftrack.loc[ii] = [track, frames, centers]

    return dftrack

def read_rectangles(df):

    dfrectangles = pd.DataFrame(columns = ['frame', 'rectangles']) # 'tid', 'x1', 'y1', 'x2', 'y2'
    frames = pd.unique(df['frameId'])

    df['tid'] = df['trackId'].astype(int) 
    df['x1'] = df['tlx'].astype(float) 
    df['y1'] = df['tly'].astype(float)
    df['x2'] = df['tlx'].astype(float) + df['width'].astype(float)
    df['y2'] = df['tly'].astype(float) + df['height'].astype(float)

    for frame in frames:
        dff = df.loc[df['frameId'] == frame, ['tid', 'x1', 'y1', 'x2', 'y2']]
        list_of_rectangles = dff[['tid', 'x1', 'y1', 'x2', 'y2']].apply(tuple, axis=1).apply(tuple, by_row=False)

        dfrectangles.loc[-1] = [frame, list_of_rectangles]
        dfrectangles.index += 1
        dfrectangles.sort_index()
    
    dfrectangles.set_index('frame')
    return dfrectangles


def plot_rectangles(img, rectangles, fr, dftrack=None):
    # Plot rectangles
    
    for rectangle in rectangles:
        x1, y1, x2, y2 = [int(x) for x in rectangle[1:]]
        tid = int(rectangle[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), colours[tid % 32], 3)

        # Draw object trail
        if type(dftrack) == pd.DataFrame:
            prev_bboxes = dftrack.loc[dftrack['trackId'] == tid, 'centers'].tolist()[0]
            frames      = dftrack.loc[dftrack['trackId'] == tid, 'frames'].tolist()[0]

            for ii in range(1, len(frames)):
                if frames[ii] <= fr:
                    c0 = tuple(prev_bboxes[ii - 1])
                    c1 = tuple(prev_bboxes[ii])

                    img = cv2.line(img, c0, c1, colours[tid % 32], thickness=3, lineType=8)
                else:
                    break

    return img


if __name__ == '__main__':
    # read arguments
    tracking_file, input_video, out_video, downsample_video, start_frame, max_frame = parse_args(sys.argv)

    with VideoCapture(input_video) as capture:

        ww = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        hh = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if downsample_video:
            ww, hh = int(np.round(ww / 2.0)), int(np.round(hh / 2.0))
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        with VideoWritter(out_video, fps, (hh, ww), "mp4v", color=True) as out:

            df = pd.read_csv(tracking_file, sep=',', header=None)
            df.columns= ['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf', 'a', 'b', 'c']

            # info about tracks init-end
            dftrack = trackinfo(df)
            # Read rectangles from text file
            rectangles = read_rectangles(df)

            fr = start_frame
            while (max_frame <= 0) or (fr < max_frame):

                if fr % 500 == 0:
                    print (f'Processing frame {fr}', file=sys.stderr)
                    
                ret, frame = capture.read()
                if frame is None:
                    break

                if fr in rectangles['frame']:
                    # Plot rectangles on image
                    frame = plot_rectangles(frame, rectangles.loc[fr, 'rectangles'], fr, dftrack)

                frame = frame.astype(np.uint8)
                if downsample_video:
                    out.write(cv2.pyrDown(frame))
                else:
                    out.write(frame)
                fr = fr + 1
                