
import os
import sys

from ceab_ants.detection.single_video_processor import SingleVideoProcessor

from docopts.help_ant_detection_yolo_bigNMS import parse_args
from input_utils.crop_frame_loader import CropFrameLoader
from models.yolo_big_area_nms import YOLO_BigAreaNMS


if __name__ == '__main__':

    queue_size = 88
    batch_size = 30
    min_batch_size = 22
    th_color = 50+1
    nms_iou = 0.5
    nms_dist = 500
    tqdm_interval = 1

    # read arguments
    input_video, detection_file, weights_path, imgsz, overlap, conf, stop_frame, initial_frame = parse_args(sys.argv)

    os.makedirs(os.path.dirname(detection_file) or '.', exist_ok=True)

    loader = CropFrameLoader(imgsz=imgsz, overlap=overlap, th_color=th_color)
    preprocess = loader.preprocess

    model = YOLO_BigAreaNMS(weights_path, imgsz, conf=conf, nms_iou=nms_iou, nms_dist=nms_dist, verbose=True)

    worker = SingleVideoProcessor(model.build_model, model.apply_model, preprocess, model.postprocess, queue_size, batch_size, min_batch_size, tqdm_interval=tqdm_interval)

    worker.process_video(input_video, detection_file, stop_frame - initial_frame, initial_frame)
