
# TODO: interface to say that the 2 IDs are the same (memory and current) keeping the previous ID so, if it recovers or changes at some point, not all the track changes.
# TODO: interface to resolve interference when 1 ID appears twice or more

import cv2
import os
import numpy as np
import queue
import threading
from tqdm import tqdm

from utils.video_player import VideoPlayerContext
from utils.frame_drawer import FrameDrawer


# INPUTS
HOMOGRAPHIES_PATH = "~/ANTS/src8_tracking/wp2_visualization/cameras_homographies.npy"
CAM0_TO_MM_PATH = "~/ANTS/src8_tracking/wp2_visualization/cam0_to_mm.npy"
MOTFILE = "~/results_video_pol/yolo_x264_stack_20240725_PINK-BLUE_WORKERS-SCOUTS_1/tracks/20240725_0941_fgbg_trk_world.txt"
BASE_CAM = 0
MAX_INPUT_BUFFER = 1
VIDEO_RESOLUTION = (4000., 3000.) # (4000., 3000.) # (500., 375.)
CAMERA_RESOLUTION = (4000., 3000.) # More like homography input resolution (which is camera full resolution)
OUTPUT_RESOLUTION = (4000., 3000.)
DEFAULT_CAMS = list(range(12))


HOMOGRAPHIES_PATH = os.path.expanduser(HOMOGRAPHIES_PATH)
CAM0_TO_MM_PATH = os.path.expanduser(CAM0_TO_MM_PATH)
MOTFILE = os.path.expanduser(MOTFILE)


def load_background_images(image_paths, drawer, resolution, num_cams):
    if image_paths: # TODO: mapa de punts relevants desde la perspectiva de cada camera
        backgrounds = []
        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (int(resolution[0]), int(resolution[1])))
                backgrounds.append((i, drawer.process_cam(img, i)))
    else:
        white = np.full((int(resolution[1]), int(resolution[0]), 3), 255, dtype=np.uint8)
        backgrounds = [(i, drawer.process_cam(white.copy(), i)) for i in range(num_cams)]

    return backgrounds


if __name__ == "__main__":
    available_cams = DEFAULT_CAMS
    background_images = []

    cam_homographies = np.load(HOMOGRAPHIES_PATH)
    real_world_homography = np.load(CAM0_TO_MM_PATH)
    selected_homographies = cam_homographies[available_cams][:, BASE_CAM, ...].reshape(-1, 3, 3)

    drawer = FrameDrawer(selected_homographies, real_world_homography, motfile=MOTFILE, video_resolution=VIDEO_RESOLUTION, output_resolution=OUTPUT_RESOLUTION, camera_resolution=CAMERA_RESOLUTION)
    background_frames = load_background_images(background_images, drawer, VIDEO_RESOLUTION, len(available_cams))
    num_frames = int(np.max(drawer.processed_mot[:, 0]))

    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    with VideoPlayerContext(frame_queue, stop_event, memory_size=1, buffer_size=2, output_resolution=OUTPUT_RESOLUTION):

        try:
            for i in tqdm(range(num_frames)):
                processed_frame = drawer.draw(background_frames, i)
                frame_queue.put(processed_frame)
        except KeyboardInterrupt:
            print("Stopping playback...")
        finally:
            stop_event.set()
            print("Playback stopped.")

