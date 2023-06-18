
from contextlib import contextmanager
import cv2 as cv
import sys
import torch
from ultralytics import YOLO

from docopts.help_ant_detection_yolo import parse_args 


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
    input_video, detection_file, weights_path, imgsz, stop_frame = parse_args(sys.argv)

    # Ensamble the Detector
    model = YOLO(weights_path, 'detect')
    model.predict(imgsz=imgsz, iou=0.7)
    def detector_model(img):
        #results = model([img])[0].boxes
        results = model([img], verbose=False)[0].boxes
        xywh = results.xywh #N, 4
        conf = results.conf.reshape(-1, 1) #N,
        bbox = torch.cat((xywh, conf), dim=1)
        return bbox


    # Apply the model
    fr = 0
    with VideoCapture(input_video) as capture:
        with open(detection_file, 'w') as out_file:
            while stop_frame <= 0 or fr <= stop_frame:
                fr = fr + 1
                
                if fr % 500 == 0:
                    print (f'Processing frame {fr}', file=sys.stderr)

                _, frame = capture.read()
                if frame is None:
                    print (f'Frame {fr} is None', file=sys.stderr)
                    break
                    
                bboxes = detector_model(frame)
                if bboxes is None:
                    continue # Training Background
                
                if len(bboxes) > 0:
                    # bbox = (x1, y1, w, h)
                    MOTDet_line = lambda fr, bbox : f'{fr}, -1, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {bbox[4]}, -1, -1, -1'
                    detection_text = '\n'.join([MOTDet_line(fr, bbox) for bbox in bboxes])
                    print (detection_text, file=out_file)
