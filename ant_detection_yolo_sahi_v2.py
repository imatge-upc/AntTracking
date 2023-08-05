
from contextlib import contextmanager
import cv2 as cv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import (
    download_yolov8s_model,
)
import sys
import torch

from docopts.help_ant_detection_yolo_sahi_v2 import parse_args 


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


if __name__ == '__main__':
    # read arguments
    input_video, detection_file, weights_path, imgsz, stop_frame, conf, initial_frame = parse_args(sys.argv)

    # Ensamble the Detector
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=weights_path,
        confidence_threshold=conf,
        device="cuda:0", # 'cpu' or 'cuda:0'
    )

    def detector_model(img):
        result = get_sliced_prediction(
            img,
            detection_model,
            slice_height = imgsz,
            slice_width = imgsz,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2,
            verbose=0
        )

        results = result.to_coco_annotations()
        xywh = torch.tensor([obj['bbox'] for obj in results]) #N, 4
        conf = torch.tensor([obj['score'] for obj in results]).reshape(-1, 1) #N,
        bbox = torch.cat((xywh, conf), dim=1)
        return bbox


    # Apply the model
    fr = initial_frame
    results = []
    with VideoCapture(input_video) as capture:
        capture.set(cv.CAP_PROP_POS_FRAMES, initial_frame - 1)
        last_frame = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) if stop_frame <= 0 else min(int(capture.get(cv.CAP_PROP_FRAME_COUNT)), initial_frame + stop_frame)

        while stop_frame <= 0 or fr < initial_frame + stop_frame:
            fr = fr + 1
            
            seen = fr - initial_frame
            if (seen == 1) or (seen == 5) or (seen == 10) or (seen == 25) or (seen == 50) or (seen % 100 == 0):
                print (f'Processing frame {fr} / {last_frame}', file=sys.stderr)

            _, frame = capture.read()
            if frame is None:
                print (f'Frame {fr} is None', file=sys.stderr)
                break
                
            bboxes = detector_model(frame)
            if bboxes is None:
                continue # Training Background
            
            if len(bboxes) > 0:
                # bbox = (x1, y1, w, h)
                MOTDet_line = lambda fr, bbox : f'{fr}, -1, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {bbox[4]}, -1, -1, -1\n'
                detection_text = ''.join([MOTDet_line(fr, bbox) for bbox in bboxes])
                results.append(detection_text)
    
    mode = 'a' if initial_frame > 1 else 'w'
    with open(detection_file, mode) as f:
        f.writelines(results)
