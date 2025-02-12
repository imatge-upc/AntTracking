
# TODO: We want to know where the models fail so we can upgrade the dataset with improved data, the first step is the retrival of error frames (either empty or to be reannotated)

import cv2
from docopt import docopt
import numpy as np
import os
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
import sys
from tqdm import tqdm

from ceab_ants.io.video_contextmanager import VideoCapture


colours = [
    (240,163,255),
    (0,117,220),
    (153,63,0),
    (76,0,92),
    (25,25,25),
    (0,92,49),
    (43,206,72),
    (255,204,153),
    (128,128,128),
    (148,255,181),
    (143,124,0),
    (157,204,0),
    (194,0,136),
    (0,51,128),
    (255,164,5),
    (255,168,187),
    (66,102,0),
    (255,0,16),
    (94,241,242),
    (0,153,143),
    (224,255,102),
    (116,10,255),
    (153,0,0),
    (255,255,128),
    (255,255,0),
    (255,80,5)
]


def seq_to_fr_boxes(seq, fr, obbox=False):
    if obbox:
        return seq[seq[:, 0] == fr, :][:, [2, 3, 4, 5, 10]]
    else:
        return seq[seq[:, 0] == fr, 2:6]


def create_obb(cx, cy, w, h, theta):
    # Define the initial rectangle
    rectangle = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
    # Rotate and translate the rectangle
    rotated_rect = rotate(rectangle, theta, use_radians=False)
    obb = translate(rotated_rect, cx, cy)
    return obb

def obbox_iou(box1, box2):
    cx1, cy1, w1, h1, theta1 = box1
    cx2, cy2, w2, h2, theta2 = box2

    obb1 = create_obb(cx1, cy1, w1, h1, theta1)
    obb2 = create_obb(cx2, cy2, w2, h2, theta2)

    intersection_area = obb1.intersection(obb2).area
    union_area = obb1.area + obb2.area - intersection_area

    iou = intersection_area / union_area
    return iou

def bbox_iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union

def error_frame(boxes_true, boxes_pred, scores, threshold=0.5, iou_funct=None):
   
    iou_funct = iou_funct or bbox_iou

    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
        
    results = {"FN" : False, "FP" : False}
    matched_bt = set()
    for bt in boxes_true:
        matched = False
        for j, bp in enumerate(boxes_pred):
            miou = iou_funct(bt, bp)
            if miou >= threshold and not matched and j not in matched_bt:
                matched = True
                matched_bt.add(j)

        if not matched:
            results["FN"] = True # bt has no match, count as FN
            
    fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
    if fp > 0:
        results["FP"] = True
    
    return results

def draw_bbox(frame, bbox, colour, thinkness=2):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    cv2.rectangle(frame, (y1, x1), (y2, x2), colour, thinkness)

    return frame

def draw_obbox(frame, bbox, colour, thinkness=2):
    cx, cy, w, h, a = bbox
    
    box = cv2.boxPoints(((cx, cy), (w, h), a))
    box = np.intp(box)

    cv2.drawContours(frame, [box], 0, colour, thinkness)

    return frame

def draw_boxes(frame, seq_dets, seq_gt, fr, obbox=False):
    frame_dets = frame.copy()
    frame_gts = frame.copy()

    for i, bbox in enumerate(seq_to_fr_boxes(seq_dets, fr, obbox=obbox)):
        colour = colours[i % len(colours)][::-1]
        if obbox:
            frame_dets = draw_obbox(frame_dets, bbox, colour)
        else:
            frame_dets = draw_bbox(frame_dets, bbox, colour)
    
    for i, bbox in enumerate(seq_to_fr_boxes(seq_gt, fr, obbox=obbox)):
        colour = colours[i % len(colours)][::-1]
        if obbox:
            frame_gts = draw_obbox(frame_gts, bbox, colour)
        else:
            frame_gts = draw_bbox(frame_gts, bbox, colour)
    
    return frame_dets, frame_gts
            


DOCTEXT = f""" 
Usage:
  error_frame_extractor.py [obbox] <detFile> <gtFile> <video> <output_folder> [--draw]
  error_frame_extractor.py -h | --help

"""

if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    det_file = args['<detFile>']
    gt_file = args['<gtFile>']
    video = args['<video>']
    folder = args['<output_folder>']

    obbox = args['obbox']
    iou_funct = obbox_iou if obbox else bbox_iou

    draw = args['--draw']

    threshold = 0.5

    os.makedirs(os.path.join(folder, "FN"), exist_ok=False)
    os.makedirs(os.path.join(folder, "FP"), exist_ok=False)

    seq_dets = np.loadtxt(det_file, delimiter=',')
    seq_gt = np.loadtxt(gt_file, delimiter=',')

    with VideoCapture(video) as capture:
        for fr in tqdm(range(len(seq_dets))):
            
            _, frame = capture.read()
            if frame is None:
                print (f'Frame {fr} is None', file=sys.stderr)
                break

            error_dict = error_frame(
                seq_to_fr_boxes(seq_gt, fr, obbox=obbox), 
                seq_to_fr_boxes(seq_dets, fr, obbox=obbox), 
                seq_dets[seq_dets[:, 0] == fr, 7], 
                thresholds=threshold, 
                iou_funct=iou_funct
            )

            frame_dets = None
            frame_gts = None

            if error_dict["FN"]:
                cv2.imwrite(os.path.join(folder, "FN", f"{fr}.png"), frame)

                if draw:
                    frame_dets, frame_gts = draw_boxes(frame, seq_dets, seq_gt, fr, obbox=obbox)
                    cv2.imwrite(os.path.join(folder, "FN", f"{fr}_det.png"), frame_dets)
                    cv2.imwrite(os.path.join(folder, "FN", f"{fr}_gt.png"), frame_gts)
            
            if error_dict["FP"]:
                cv2.imwrite(os.path.join(folder, "FP", f"{fr}.png"), frame)

                if draw:
                    if (frame_dets is None) or (frame_gts is None):
                        frame_dets, frame_gts = draw_boxes(frame, seq_dets, seq_gt, fr, obbox=obbox)
                    cv2.imwrite(os.path.join(folder, "FP", f"{fr}_det.png"), frame_dets)
                    cv2.imwrite(os.path.join(folder, "FP", f"{fr}_gt.png"), frame_gts)
