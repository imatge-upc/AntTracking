
# Adaptat del codi de https://www.kaggle.com/code/chenyc15/mean-average-precision-metric
# Amb codi de https://medium.com/mindkosh/calculating-iou-between-oriented-bounding-boxes-c39f72602cac

from docopt import docopt
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
import sys
from tqdm import tqdm


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

def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], iou_funct=None):
   
    iou_funct = iou_funct or bbox_iou

    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn, tn = 0, 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou_funct(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
                elif miou < t:
                    tn += 1
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    #total = len(boxes_pred) + len(matched_bt)
    return map_total / len(thresholds), fn, fp, tn, tp

DOCTEXT = f""" 
Usage:
  map_detections.py [obbox] <detFile> <gtFile>
  map_detections.py -h | --help

"""

if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    det_file = args['<detFile>']
    gt_file = args['<gtFile>']

    obbox = args['obbox']
    iou_funct = obbox_iou if obbox else bbox_iou

    thresholds5095 = np.arange(0.5, 1., 0.05)
    thresholds50 = [0.5]

    seq_dets = np.loadtxt(det_file, delimiter=',')
    seq_gt = np.loadtxt(gt_file, delimiter=',')

    frames = np.unique(np.concatenate((seq_dets[:, 0], seq_gt[:, 0]), axis=None))

    map50 = list()
    map5095 = list()
    fn = list()
    fp = list()
    tn = list()
    tp = list()
    for fr in tqdm(frames):
        img_map, img_fn, img_fp, img_tn, img_tp = map_iou(seq_to_fr_boxes(seq_gt, fr, obbox=obbox), seq_to_fr_boxes(seq_dets, fr, obbox=obbox), seq_dets[seq_dets[:, 0] == fr, 7], thresholds=thresholds50, iou_funct=iou_funct)
        map50.append(img_map)
        fn.append(img_fn)
        fp.append(img_fp)
        tn.append(img_tn)
        tp.append(img_tp)

        img_map, _, _, _, _ = map_iou(seq_to_fr_boxes(seq_gt, fr, obbox=obbox), seq_to_fr_boxes(seq_dets, fr, obbox=obbox), seq_dets[seq_dets[:, 0] == fr, 7], thresholds=thresholds5095, iou_funct=iou_funct)
        map5095.append(img_map)

    print(f'mAP@50 = {np.mean(map50):.02f}')
    print(f'mAP@50-95 = {np.mean(map5095):.02f}')
    print(f'FN@50 = {sum(fn)} ({sum(fn) / (sum(fn) + sum(tp)) * 100:02.02f} %)')
    print(f'FP@50 = {sum(fp)} ({sum(fp) / (sum(fp) + sum(tn)) * 100:02.02f} %)')
    print(f'Precision@50 = {sum(tp) / (sum(tp) + sum(fp)):.02f}')
    print(f'Recall@50 = {sum(tp) / (sum(tp) + sum(fn)):.02f}')
