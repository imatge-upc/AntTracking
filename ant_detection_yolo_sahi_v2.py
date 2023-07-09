
from sahi.predict import predict
from sahi.utils.yolov8 import (
    download_yolov8s_model,
)
import sys
import torch

from docopts.help_ant_detection_yolo_sahi_v2 import parse_args 


if __name__ == '__main__':
    # read arguments
    input_video, detection_file, weights_path, imgsz, stop_frame, conf = parse_args(sys.argv)

    def detector_model(input_video):
        result = predict(
            model_type='yolov8', # one of 'yolov5', 'mmdet', 'detectron2'
            model_path=weights_path, # path to model weight file
            model_confidence_threshold=conf,
            model_device='cpu', # or 'cuda:0'
            source=input_video, # image, folder or video path
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
    fr = 0
    results = []
    
    bboxes = detector_model(input_video)
           
    if len(bboxes) > 0:
        # bbox = (x1, y1, w, h)
        MOTDet_line = lambda fr, bbox : f'{fr}, -1, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {bbox[4]}, -1, -1, -1\n'
        detection_text = ''.join([MOTDet_line(fr, bbox) for bbox in bboxes])
        results.append(detection_text)
    
    with open(detection_file, 'w') as f:
        f.writelines(results)
