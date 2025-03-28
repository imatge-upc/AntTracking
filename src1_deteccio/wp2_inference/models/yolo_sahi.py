
import cv2
import numpy as np
import torch
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

class YOLO_SAHI():
    
    def __init__(self, model_path, imgsz=640, overlap=0.2, conf=0.3, verbose=True):
        self.model_path = model_path
        self.imgsz = imgsz
        self.overlap = overlap
        self.conf = conf

        self.verbose = verbose

    def build_model(self, device=None):

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.conf,
            device=device
        )

        if self.verbose : print(f"SAHI model loaded on {model.device}")

        return model

    def apply_model(self, model, batch):
        results = [
            get_sliced_prediction(
                image,
                model,
                slice_height=self.imgsz,
                slice_width=self.imgsz,
                overlap_height_ratio=self.overlap,
                overlap_width_ratio=self.overlap,
                postprocess_type="NMS", # ['GREEDYNMM', 'NMM', 'NMS', 'LSNMS']
                verbose=False,
                perform_standard_pred=False,
            ) for image in batch
        ]
        return results

    def postprocess(self, model_output, metadata):
        
        fr = metadata[1]
        
        obboxes = []
        for result in model_output:
            for det in result.object_prediction_list:
                if det.mask is not None:
                    obb_candidates = []

                    for polygon in det.mask.segmentation:
                        if len(polygon) < 6:
                            continue
                        
                        polygon_np = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                        rect = cv2.minAreaRect(polygon_np)
                        (cx, cy), (w, h), angle = rect

                        if w < h:
                            w, h = h, w
                            angle += 90  

                        obb_candidates.append((cx, cy, w, h, angle))

                    if obb_candidates:
                        cx, cy, w, h, angle = max(obb_candidates, key=lambda x: x[2] * x[3])
                    else:
                        continue
                else:
                    x_min, y_min, width, height = det.bbox.to_coco()
                    cx, cy = x_min + width / 2, y_min + height / 2
                    w, h = width, height
                    angle = 0

                confidence = det.score.value            
                obboxes.append([cx, cy, w, h, angle, confidence])

        return (fr, obboxes)
