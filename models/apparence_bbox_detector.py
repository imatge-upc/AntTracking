
import numpy as np
import torch


class ApparenceBBoxDetector():
    # NOTE: Maybe a cache can be used to apply the same model for detection and apparence

    def crop_pad(self, frame, bbox, background_color):

        img = np.full((self.height, self.width, len(background_color)), background_color).reshape(len(background_color), self.height, self.width)
        
        h = int(min(bbox[3], self.height))
        w = int(min(bbox[2], self.width))

        start_h = int(self.height // 2 - h // 2)
        start_w = int(self.width // 2 - w // 2)
        img[:, start_h : start_h + h, start_w : start_w + w] = frame[int(bbox[1]) : int(bbox[1]) + h, int(bbox[0]) : int(bbox[0]) + w, :].reshape(len(background_color), h, w)
        
        return img
    
    def __init__(self, bbox_detector, apparence_model, height=224, width=224, skip=0):
        self.bbox_detector = bbox_detector
        self.apparence_model = apparence_model

        self.height = height
        self.width = width

        self.skip = skip
    
    def apply(self, frame):
        
        bboxes = self.bbox_detector(frame) # list of (x, y, w, h)
        if bboxes is None or len(bboxes) == 0 : return [] # Previously None

        background_color = np.mean(frame, (0, 1))
        inputs = torch.Tensor(np.stack([self.crop_pad(frame, bbox[self.skip:], background_color) for bbox in bboxes], axis=0))
        outputs = self.apparence_model(inputs) # Consider [#bbox, #features] shape

        a_bboxes = [(*bbox, *desc) for bbox, desc in zip(bboxes, outputs)]

        return a_bboxes

    __call__ = apply
