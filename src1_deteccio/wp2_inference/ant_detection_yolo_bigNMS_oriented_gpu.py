
import cv2
from functools import lru_cache
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ceab_ants.detection.utils.nms import bigAreaOneClassNMS, bigAreaOneClassMaskedNMS, OBBox
from ceab_ants.detection.utils.sliding_windows import sliceFrame
from ceab_ants.io.video_contextmanager import VideoCapture

from docopts.help_ant_detection_yolo_bigNMS import parse_args


# TODO: adapt to process_video

@lru_cache(maxsize=1)
def compute_overlap_mask(height, width, imgsz, overlap):
    overlap_mask = np.zeros((height, width), dtype=bool)
    stride_h, stride_w = (imgsz * (1 - overlap)).astype(int)
    for y_offset in range(0, height - imgsz, stride_h):
        overlap_mask[y_offset + imgsz - stride_h:y_offset + imgsz, :] = True
    for x_offset in range(0, width - imgsz, stride_w):
        overlap_mask[:, x_offset + imgsz - stride_w:x_offset + imgsz] = True
    return overlap_mask

def get_obbox(det):
    return det[:-1] # x, y, w, h, a, s

if __name__ == '__main__':
    # read arguments
    input_video, detection_file, weights_path, imgsz, overlap, conf, stop_frame, initial_frame = parse_args(sys.argv)

    os.makedirs(os.path.dirname(detection_file) or '.', exist_ok=True)

    # Ensamble the Detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detection_model = YOLO(weights_path)
    detection_model.to(device)

    def detector_model(img):
        height, width = img.shape[:2]
        rgb_img = img[..., ::-1] if len(img.shape) == 3 else img
        crops, offsets = sliceFrame(rgb_img, imgsz, overlap, batch=True)

        overlap_mask = compute_overlap_mask(height, width, imgsz, overlap)

        results = detection_model(crops, imgsz=imgsz, conf=conf, verbose=False)

        bboxes = []
        for result, offset in zip(results, offsets):
            xywhr = result.obb.xywhr.cpu().reshape(-1, 5) # N', 5
            score = result.obb.conf.cpu().reshape(-1, 1) # N', 1
            xywhr[:, -1] = torch.rad2deg(xywhr[:, -1])

            bad = (xywhr[:, 0] - xywhr[:, 2] / 2 <= 0) | (xywhr[:, 1] - xywhr[:, 3] / 2 <= 0) | (xywhr[:, 0] + xywhr[:, 2] / 2 >= imgsz) | (xywhr[:, 1] + xywhr[:, 3] / 2 >= imgsz) 
            xywhr, score = xywhr[~bad, :], score[~bad, :]

            xywhr[:, 0] = xywhr[:, 0] + offset[1]
            xywhr[:, 1] = xywhr[:, 1] + offset[0]

            bboxes.append(torch.cat((xywhr, score), dim=1)) # B, N', 5

        bboxes = torch.cat(bboxes, dim=0) # N, 5
        bad = (bboxes[:, 0] + bboxes[:, 2] / 2 > width) | (bboxes[:, 1] + bboxes[:, 3] / 2 > height)
        bboxes = bboxes[~bad, :]

        #nms_bboxes = bigAreaOneClassNMS(bboxes.tolist(), th_iou=0.5, max_distance=500, get_bbox_funct=get_obbox, bbox_class=OBBox) # N, 5
        nms_bboxes = bigAreaOneClassMaskedNMS(bboxes.tolist(), overlap_mask, th_iou=0.5, max_distance=500, get_bbox_funct=get_obbox, bbox_class=OBBox)

        return nms_bboxes # N, 5

    # Apply the model
    fr = initial_frame
    results = []
    with VideoCapture(input_video) as capture:
        capture.set(cv2.CAP_PROP_POS_FRAMES, initial_frame - 1)
        max_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        last_frame = max(max_frame, 0) if stop_frame <= 0 else min(max(max_frame, -max_frame), initial_frame + stop_frame)

        mode = 'a' if initial_frame > 1 else 'w'
        with open(detection_file, mode) as out_file: # open can use buffering= to set the buffer size
            # For large files it is better to keep it open
            
            def generator(fr):
                while stop_frame <= 0 or fr < initial_frame + stop_frame:
                    yield

            for _ in tqdm(generator(fr), mininterval=10, maxinterval=10):
                fr = fr + 1

                # TODO: Modify to batch more frames at once (now: 5h 39min 45s for 9014 frames -> x12 -> 2 days 19h 57min + tracking per 20 min 20 formigues)
                # TODO: Solve dataset generation, min size ant after obbox crop
                
                seen = fr - initial_frame
                #if (seen == 1) or (seen == 5) or (seen == 10) or (seen == 25) or (seen == 50) or (seen % 100 == 0):
                #    print (f'Processing frame {fr} / {last_frame}', file=sys.stderr)

                _, frame = capture.read()
                if frame is None:
                    print (f'Frame {fr} is None', file=sys.stderr)
                    break
                    
                obboxes = detector_model(frame)
                if obboxes is None:
                    continue # Training Background
                
                if len(obboxes) > 0:
                    # bbox = (x, y, w, h, r, s) -> (fr, -1, x, y, w, h, s, -1, -1, -1, r)
                    MOTDet_line = lambda fr, obbox : f'{fr:.0f},-1,{obbox[0]:.5f},{obbox[1]:.5f},{obbox[2]:.5f},{obbox[3]:.5f},{obbox[5]:.5f},-1,-1,-1,{obbox[4]:.1f}'
                    detection_text = '\n'.join([MOTDet_line(fr, bbox) for bbox in obboxes])
                    print(detection_text, end='\n', file=out_file)
