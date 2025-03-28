
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLO_Resize():
    
    def __init__(self, model_path, conf=0.3, verbose=True):
        self.model_path = model_path
        self.conf = conf

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
            results = model.predict(batch, conf=self.conf, stream=True, verbose=False)
        return [result.cpu() for result in results]

    def postprocess(self, model_output, metadata):
        
        fr = metadata[1]
        
        obboxes = []
        for result in model_output:

            if result.obb is not None:
                xywhr = result.obb.xywhr.cpu().numpy().reshape(-1, 5)
                confidences = result.obb.conf.cpu().numpy().reshape(-1, 1)
                xywhr[:, -1] = np.rad2deg(xywhr[:, -1])
                obboxes = np.concatenate((xywhr, confidences), axis=1)

            elif result.masks is not None:
                masks = result.masks.cpu().xy
                scores= result.boxes.conf.cpu()
                for polygon, score in zip(masks, scores.numpy()):
                    polygon_np = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                    rect = cv2.minAreaRect(polygon_np)
                    (cx, cy), (w, h), angle = rect

                    if w < h:
                        w, h = h, w
                        angle += 90

                    obboxes.append([cx, cy, w, h, angle, score])

        return (fr, obboxes)
