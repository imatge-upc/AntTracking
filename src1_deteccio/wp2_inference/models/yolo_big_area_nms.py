
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ceab_ants.detection.utils.nms import bigAreaOneClassMaskedNMS, OBBox


class YOLO_BigAreaNMS():
    
    def __init__(self, model_path, imgsz=640, conf=0.3, nms_iou=0.5, nms_dist=500, verbose=True):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.nms_iou = nms_iou
        self.nms_dist = nms_dist

        self.verbose = verbose

    def build_model(self, device=None):

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        model = YOLO(self.model_path)
        model.to(device)

        if self.verbose : print(f"model built on {device}")

        return model

    def apply_model(self, model, batch):
        with torch.no_grad():
            results = model.predict(batch, imgsz=self.imgsz, conf=self.conf, stream=True, verbose=False)
        return [result.cpu() for result in results]

    def postprocess(self, model_output, metadata):
        
        fr = metadata[1]
        imgsz = metadata[3]
        offsets = metadata[4]
        overlap_mask = metadata[5]
        height = metadata[6]
        width = metadata[7]

        bboxes = []
        for result, offset in zip(model_output, offsets):
            if result.obb is not None:
                xywhr = result.obb.xywhr.cpu().reshape(-1, 5) # N', 5
                score = result.obb.conf.cpu().reshape(-1, 1) # N', 1
                xywhr[:, -1] = torch.rad2deg(xywhr[:, -1])

                bad = (xywhr[:, 0] - xywhr[:, 2] / 2 <= 0) | (xywhr[:, 1] - xywhr[:, 3] / 2 <= 0) | (xywhr[:, 0] + xywhr[:, 2] / 2 >= imgsz) | (xywhr[:, 1] + xywhr[:, 3] / 2 >= imgsz) 
                xywhr, score = xywhr[~bad, :], score[~bad, :]

                xywhr[:, 0] = xywhr[:, 0] + offset[1]
                xywhr[:, 1] = xywhr[:, 1] + offset[0]

                bboxes.append(torch.cat((xywhr, score), dim=1)) # B, N', 5
            elif result.masks is not None:
                b_bboxes = []

                masks = result.masks.cpu().xy
                scores = result.boxes.conf.cpu()
                for polygon, score in zip(masks, scores.numpy()):
                    polygon_np = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                    rect = cv2.minAreaRect(polygon_np)
                    (cx, cy), (w, h), angle = rect

                    if w < h:
                        w, h = h, w
                        angle += 90

                    bad = ((cx - w / 2 <= 0) | (cy - h / 2 <= 0) | (cx + w / 2 >= imgsz) | (cy + h / 2 >= imgsz))
                    if not bad:
                        b_bboxes.append(torch.as_tensor([cx + offset[1], cy + offset[0], w, h, angle, score]).reshape(-1, 6))

                if b_bboxes:
                    bboxes.append(torch.cat(b_bboxes, dim=0))  # B, N', 5

        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)  # N, 5
        else:
            bboxes = torch.empty((0, 5))
            return (fr, bboxes)
            
        bad = (bboxes[:, 0] + bboxes[:, 2] / 2 > width) | (bboxes[:, 1] + bboxes[:, 3] / 2 > height)
        bboxes = bboxes[~bad, :]

        def get_obbox(det):
            return det[:-1] # x, y, w, h, a, s
        nms_bboxes = bigAreaOneClassMaskedNMS(bboxes.tolist(), overlap_mask, th_iou=self.nms_iou, max_distance=self.nms_dist, get_bbox_funct=get_obbox, bbox_class=OBBox)

        return (fr, nms_bboxes)
