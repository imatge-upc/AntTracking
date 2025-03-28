
# TODO: interface to say that the 2 IDs are the same (memory and current) keeping the previous ID so, if it recovers or changes at some point, not all the track changes.
# TODO: interface to resolve interference when 1 ID appears twice or more

from docopt import docopt
import glob
import json
import numpy as np
import os
import queue
import sys
import threading
from tqdm import tqdm

from ceab_ants.io.frame_loader_sync import SyncFrameLoaderContext

from utils.video_player import VideoPlayerContext
from utils.frame_drawer import FrameDrawer


def main(motfile, video_files, cam_homographies, real_world_homography, input_resolution, output_resolution, fps=7.5):

    CAMERA_RESOLUTION = (4000., 3000.) # Resolution for the homographies
    MAX_INPUT_BUFFER = 3

    MAX_IDS = None
    #COLORMAP_FILE = None
    #MAX_IDS = 100
    COLORMAP_FILE = "~/ANTS/src8_tracking/wp2_visualization/100_colors.json"
    COLORMAP_FILE = os.path.expanduser(COLORMAP_FILE) if COLORMAP_FILE is not None else None

    available_cams = [int(f[-6:-4]) for f in video_files]                           # NOTE: using the regex, the cam number extraction will be more stable

    selected_homographies = cam_homographies[available_cams][:, ...].reshape(-1, 3, 3)

    colormap = None
    if COLORMAP_FILE is not None:
        colormap = []
        with open(COLORMAP_FILE) as json_file:
            colormap = json.load(json_file)

        if MAX_IDS is None:
            mot = np.loadtxt(motfile, delimiter=',')
            MAX_IDS = len(np.unique(mot[:, 1]))
        
        if mot[0, 1] != -1 and len(colormap) > 0:
            while MAX_IDS > len(colormap):
                colormap.extend(colormap.copy())
            colormap = colormap[:MAX_IDS]
        
        colormap = {i : c for i, c in enumerate(colormap.copy())}

    drawer = FrameDrawer(selected_homographies, real_world_homography, motfile=motfile, memory_size=100, 
                         video_resolution=input_resolution, output_resolution=output_resolution, 
                         camera_resolution=CAMERA_RESOLUTION, colormap=colormap, max_ids=MAX_IDS)

    with SyncFrameLoaderContext(video_files, preprocess_func=drawer.process_cam, max_buffer=MAX_INPUT_BUFFER) as frame_loader:

        frame_queue = queue.Queue(maxsize=10)
        stop_event = threading.Event()
        with VideoPlayerContext(frame_queue, stop_event, memory_size=1, buffer_size=2, output_resolution=output_resolution):

            try:
                for i, frames in tqdm(enumerate(frame_loader.frames_generator())):
                    processed_frame = drawer.draw(frames, i)
                    frame_queue.put(processed_frame)
            except KeyboardInterrupt:
                print("Stopping playback...")
            finally:
                stop_event.set()
                print("Playback stopped.")


DOCTEXT = """
Video processing and visualization with detections and tracks.

Usage:
  play.py <video_dir> <motfile> <homographies> [--fps=<fps>] [--output_resolution=<res>] [--input_resolution=<res>] [--real_world_homography=<rwh>]
  play.py --single_video <video_dir> <motfile> [--fps=<fps>] [--output_resolution=<res>] [--input_resolution=<res>] [--real_world_homography=<rwh>]

Options:
  --fps=<fps>                      Frames per second for the output video [default: 7.5].
  --output_resolution=<res>        Output video resolution (width,height) [default: 4000,4000].
  --input_resolution=<res>         Input video resolution (width,height) [default: 4000,3000].
  --real_world_homography=<rwh>    Path to the .npy to translate from joined plane to real world coordinates.
"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video_dir = args['<video_dir>']
    motfile = args['<motfile>']
    fps = float(args['--fps'])
    input_resolution = tuple(map(int, args['--input_resolution'].split(',')))
    output_resolution = tuple(map(int, args['--output_resolution'].split(',')))
    real_world_homography_file = args["--real_world_homography"]
    
    if args['--single_video']:
        homographies_path = None
        cam_homographies = np.eye(3).reshape(1, 1, 3, 3)
        video_files = [video_dir]
    else:
        homographies_path = args['<homographies>']
        cam_homographies = np.load(homographies_path)
        video_files = sorted(glob.glob(os.path.join(video_dir, "*_[0-1][0-9].mkv")))    # NOTE: The regex for the files will be an input later
        if len(video_files) == 0:
            video_files = sorted(glob.glob(os.path.join(video_dir, "*_[0-1][0-9].mp4")))

    real_world_homography = np.load(real_world_homography_file) if real_world_homography_file is not None else np.eye(3)

    main(motfile, video_files, cam_homographies, real_world_homography, input_resolution, output_resolution, fps=fps)

