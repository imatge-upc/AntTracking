
import cv2
import numpy as np


class PcaBBoxDetector():

    def crop_pca_orientation(self, frame, bbox, background_th):
        crop = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
        pts = np.argwhere(crop < background_th).reshape(-1, 2).astype(np.float32)
        if len(pts) < self.min_size : return 0

        mean = np.empty((0))
        _, eigenvectors, _ = cv2.PCACompute2(pts, mean)

        angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
        return angle

    def __init__(self, bbox_detector, background_th=None, min_size=20):
        self.bbox_detector = bbox_detector

        self.background_th = background_th
        self.min_size = min_size
    
    def apply(self, frame):
        
        bboxes = self.bbox_detector(frame)
        if bboxes is None : return None
        
        gray_frame = len(bboxes) == 0 or cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_th = self.background_th or len(bboxes) == 0 or np.mean(gray_frame) * 0.5

        angles = np.asarray([self.crop_pca_orientation(gray_frame, bbox, background_th) for bbox in bboxes]).reshape(-1, 1)

        bboxes_pca = np.hstack((bboxes, angles))
        return bboxes_pca

    __call__ = apply