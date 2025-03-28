
from docopt import docopt
import glob
import json
import numpy as np
import os
import queue
import sys
import threading
import time
from tqdm import tqdm

from ceab_ants.io.frame_loader_sync import SyncFrameLoaderContext

from utils.video_writer import VideoWriterContext
from utils.frame_drawer import FrameDrawer


def main(motfile, video_files, cam_homographies, output_video, real_world_homography, input_resolution, output_resolution, fps=7.5):

    CAMERA_RESOLUTION = (4000., 3000.) # Resolution for the homographies
    MAX_INPUT_BUFFER = 3
    INI_FRAME = 0
    NUM_FRAMES = -1

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
        with VideoWriterContext(output_video, frame_queue, stop_event, fps=fps, resolution=output_resolution) as video_writer:
            print(f"Started {type(video_writer)}")
            
            try:
                for i, frames in tqdm(enumerate(frame_loader.frames_generator())):

                    if i < INI_FRAME:
                        continue

                    if NUM_FRAMES > 0 and i >= NUM_FRAMES:
                        break
                    
                    processed_frame = drawer.draw(frames, i)
                    frame_queue.put(processed_frame)
                    
            except KeyboardInterrupt:
                print("Stopping writer...")
            else:
                print("Writting last frames.")
            finally:
                stop_event.set()
                print("5 seconds to ensure VideoWriter is released.")
                time.sleep(5)
                print("Writting stopped.")


DOCTEXT = """
Video processing and visualization with detections and tracks.

Usage:
  save.py <video_dir> <motfile> <output_video> <homographies> [--fps=<fps>] [--output_resolution=<res>] [--input_resolution=<res>] [--real_world_homography=<rwh>]
  save.py --single_video <video_dir> <motfile> <output_video> [--fps=<fps>] [--output_resolution=<res>] [--input_resolution=<res>] [--real_world_homography=<rwh>]

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
    output_video = args['<output_video>']
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

    main(motfile, video_files, cam_homographies, output_video, real_world_homography, input_resolution, output_resolution, fps=fps)
