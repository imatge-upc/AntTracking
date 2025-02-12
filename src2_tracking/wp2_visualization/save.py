
import glob
import json
import numpy as np
import os
import queue
import threading
import time
from tqdm import tqdm

from ceab_ants.io.sync_frame_loader import SyncFrameLoaderContext
from utils.video_writer import VideoWriterContext

from utils.frame_drawer import FrameDrawer

# INPUTS
VIDEO_DIR = "~/x264_stack_20240725_PINK-BLUE_WORKERS-SCOUTS_1/resize/"
HOMOGRAPHIES_PATH = "~/ANTS/src8_tracking/wp2_visualization/cameras_homographies.npy"
CAM0_TO_MM_PATH = "~/ANTS/src8_tracking/wp2_visualization/cam0_to_mm.npy"
MOTFILE = "~/results_video_pol/yolo_x264_stack_20240725_PINK-BLUE_WORKERS-SCOUTS_1/tracks/20240725_0941_fgbg_trk_world.txt"
BASE_CAM = 0
MAX_INPUT_BUFFER = 1
VIDEO_RESOLUTION = (500., 375.)
CAMERA_RESOLUTION = (4000., 3000.)
OUTPUT_RESOLUTION = (4000., 4000.)
OUTPUT_VIDEO_PATH = "/home/ignasi/ANTS/src8_tracking/wp2_visualization/output.mp4"
FPS = 7.5
INI_FRAME = 0
NUM_FRAMES = -1

MAX_IDS = None
#COLORMAP_FILE = None
#MAX_IDS = 100
COLORMAP_FILE = "/home/ignasi/ANTS/src8_tracking/wp2_visualization/100_colors.json"

VIDEO_DIR = os.path.expanduser(VIDEO_DIR)
HOMOGRAPHIES_PATH = os.path.expanduser(HOMOGRAPHIES_PATH)
CAM0_TO_MM_PATH = os.path.expanduser(CAM0_TO_MM_PATH)
MOTFILE = os.path.expanduser(MOTFILE)
COLORMAP_FILE = os.path.expanduser(COLORMAP_FILE) if COLORMAP_FILE is not None else None


if __name__ == "__main__":

    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_[0-1][0-9].mkv")))
    available_cams = [int(f[-6:-4]) for f in video_files]

    cam_homographies = np.load(HOMOGRAPHIES_PATH)
    real_world_homography = np.load(CAM0_TO_MM_PATH)
    selected_homographies = cam_homographies[available_cams][:, BASE_CAM, ...].reshape(-1, 3, 3)

    colormap = None
    if COLORMAP_FILE is not None:
        colormap = []
        with open(COLORMAP_FILE) as json_file:
            colormap = json.load(json_file)

        if MAX_IDS is None:
            mot = np.loadtxt(MOTFILE, delimiter=',')
            MAX_IDS = len(np.unique(mot[:, 1]))
        
        if mot[0, 1] != -1 and len(colormap) > 0:
            while MAX_IDS > len(colormap):
                colormap.extend(colormap.copy())
            colormap = colormap[:MAX_IDS]
        
        colormap = {i : c for i, c in enumerate(colormap.copy())}

    drawer = FrameDrawer(selected_homographies, real_world_homography, motfile=MOTFILE, memory_size=100, 
                         video_resolution=VIDEO_RESOLUTION, output_resolution=OUTPUT_RESOLUTION, 
                         camera_resolution=CAMERA_RESOLUTION, colormap=colormap, max_ids=MAX_IDS)

    with SyncFrameLoaderContext(video_files, preprocess_func=drawer.process_cam, max_buffer=MAX_INPUT_BUFFER) as frame_loader:
        
        frame_queue = queue.Queue(maxsize=10)
        stop_event = threading.Event()
        with VideoWriterContext(OUTPUT_VIDEO_PATH, frame_queue, stop_event, fps=FPS, resolution=OUTPUT_RESOLUTION) as video_writer:
            
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
