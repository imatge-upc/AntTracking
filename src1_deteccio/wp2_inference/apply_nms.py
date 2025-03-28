
import cv2
from docopt import docopt
import numpy as np
import sys
from tqdm import tqdm

from ceab_ants.detection.utils.nms import bigAreaOneClassNMS, bigAreaOneClassMaskedNMS, get_obbox, OBBox
from ceab_ants.io.mot_loader import MOTLoader


DOCTEXT = f"""
Usage:
  apply_nms.py <motfile> <output> [--iou=<i>] [--max_dist=<md>] [--mask=<m>]
  apply_nms.py oriented <motfile> <output> [--iou=<i>] [--max_dist=<md>] [--mask=<m>]

Options:
  --iou=<i>          Thereshold on bbox supression [default: 0.5]
  --max_dist=<md>    Maximum center distance to compute IoU [default: 50]
  --mask=<m>         Path to the mask file
"""

if __name__ == "__main__":
    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    
    motfile = args['<motfile>']
    output = args['<output>']

    th_iou = float(args['--iou'])
    max_distance = float(args['--max_dist'])

    oriented = args['oriented']
    mask_path = args['--mask']

    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    detections = MOTLoader(motfile)

    with open(output, 'w') as out_file:
        for fr in tqdm(range(1, detections.system_last_frame + 1)):
            dets = detections(fr)
            if dets.size == 0 : continue

            get_bbox_funct, bbox_class = (get_obbox, OBBox) if oriented else (None, None)
            
            if mask is not None:
                nms_bboxes = bigAreaOneClassMaskedNMS(dets, mask, th_iou=th_iou, max_distance=max_distance, get_bbox_funct=get_bbox_funct, bbox_class=bbox_class)
            else:
                nms_bboxes = bigAreaOneClassNMS(dets, th_iou=th_iou, max_distance=max_distance, get_bbox_funct=get_bbox_funct, bbox_class=bbox_class)

            MOTDet_line = lambda bbox : f'{int(bbox[0])}, {int(bbox[1])}, {", ".join([f"{b:.5f}" for b in bbox[2:]])}'
            results = '\n'.join([MOTDet_line(bbox) for bbox in nms_bboxes])
            print(results, file=out_file)
