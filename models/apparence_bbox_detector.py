
import cv2 as cv
import numpy as np
import torch


def crop_pad(crop, bbox, crop_w, crop_h, pad_color):

    h = bbox[3]
    w = bbox[2]
    
    if h > crop_h:
        excess = h - crop_h
        crop = crop[excess // 2 : excess // 2 + crop_h, :, :]
        h = crop_h
    
    if w > crop_w:
        excess = w - crop_w
        crop = crop[:, excess // 2 : excess // 2 + crop_w, :]
        w = crop_w
    
    if h < crop_h or w < crop_w:
        pad_h = (crop_h - h) // 2
        pad_w = (crop_w - w) // 2
        pad = ((pad_h, crop_h - h - pad_h), (pad_w, crop_w - w - pad_w))
        
        crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    
    crop = np.moveaxis(crop, [0, 1, 2], [1, 2, 0])
    return crop

def pad_reshape(crop, bbox, crop_w, crop_h, pad_color):

    h = bbox[3]
    w = bbox[2]

    ar = crop_h / crop_w
    
    pad_h = int((w * ar - h) // 2)
    pad_w = int((h / ar - w) // 2)
    pad = ((pad_h, int(w * ar - h - pad_h)), (0, 0)) if h < w * ar else ((0, 0), (pad_w, int(h / ar - w - pad_w)))
    
    crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    
    crop = cv.resize(crop, (crop_w, crop_h), interpolation=cv.INTER_AREA)

    crop = np.moveaxis(crop, [0, 1, 2], [1, 2, 0])
    return crop


class ApparenceBBoxDetector():
    # NOTE: Maybe a cache can be used to apply the same model for detection and apparence

    def crop_pad(self, frame, bbox, background_color):

        h = int(bbox[3])
        w = int(bbox[2])

        crop = frame[int(bbox[1]) : int(bbox[1]) + h, int(bbox[0]) : int(bbox[0]) + w, :]

        return crop_pad(crop, bbox, self.width, self.height, background_color)
    
    def pad_reshape(self, frame, bbox, background_color):

        h = int(bbox[3])
        w = int(bbox[2])

        crop = frame[int(bbox[1]) : int(bbox[1]) + h, int(bbox[0]) : int(bbox[0]) + w, :]

        return pad_reshape(crop, bbox, self.width, self.height, background_color)
    
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
        inputs = torch.Tensor(np.stack([self.pad_reshape(frame, bbox[self.skip:], background_color) for bbox in bboxes], axis=0))
        outputs = self.apparence_model(inputs) # Consider [#bbox, #features] shape

        a_bboxes = [(*bbox, *desc) for bbox, desc in zip(bboxes, outputs)]

        return a_bboxes

    __call__ = apply
