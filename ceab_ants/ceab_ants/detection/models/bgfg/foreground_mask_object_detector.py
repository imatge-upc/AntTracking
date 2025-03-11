
import cv2
import numpy as np


def do_nothing(*args, **kargs):
    return *args, kargs

def build_bg_MOG2(var_thresh):
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=var_thresh, detectShadows=False)
    return backSub

def build_bg_KNN():
    backSub = cv2.createBackgroundSubtractorKNN()
    return backSub

def bboxes_from_conected_components(fgMask, min_size=20, connectivity=4, ltype=cv2.CV_32S):
    analysis = cv2.connectedComponentsWithStats(fgMask, connectivity, ltype)
    (totalLabels, labels, stats, centroids) = analysis

    bboxes = []
    for stat in stats:
        # extract the connected component statistics for the current label
        x = stat[cv2.CC_STAT_LEFT]
        y = stat[cv2.CC_STAT_TOP]
        w = stat[cv2.CC_STAT_WIDTH]
        h = stat[cv2.CC_STAT_HEIGHT]
        
        if w > min_size and h > min_size:
            bboxes.append((x, y, w, h))
    
    return np.asarray(bboxes).reshape(-1, 4)

def oriented_bboxes_from_conected_components(fgMask, min_size=20):
    contours, _ = cv2.findContours(fgMask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    bboxes = []
    for contour in contours:
        (x, y), (w, h), angle = cv2.minAreaRect(contour)

        if w < min_size and h < min_size:
            continue
            
        # cv2.minAreaRect give the (possitive) angle of the right vertical edge of the unrotated bbox towards the rotated one (from 0º to 90º)
        if w > h:
            # We want the angle of the direction of the ant head (without tracking or supervision, it can be the oposite direction)
            # The obb format will be the center (x, y), the dimensions (h, w) with h < w and the angle from -180º to 180º (this code will produce from -90º to 90º)
            angle = -angle
            w, h = h, w
        else:
            angle = 90 - angle

        bboxes.append((x, y, w, h, angle))
    return np.asarray(bboxes).reshape(-1, 5)


class MorphologyNoiseFilter():
    # Remove noise  https://stackoverflow.com/questions/30369031/remove-spurious-small-islands-of-noise-in-an-image-python-opencv

    def __init__(self, close_kernel=5, open_kernel=2):
        self.se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        self.se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel))
    
    def __call__(self, fgMask):
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, self.se1)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN,  self.se2)
        return fgMask

class ForegroundMaskObjectDetector():

    def __init__(self, backSub_func, bbox_extractor, filter_fg_func=None):
        self.backSub = backSub_func
        self.bbox_extractor = bbox_extractor
        self.filter_fg = filter_fg_func or do_nothing

    def train(self, frame, lr=-1):
        self.backSub.apply(frame, learningRate=lr)

    def save_background(self, filename):
        background = self.backSub.getBackgroundImage()
        cv2.imwrite(filename, background)

    def apply(self, frame, lr=0):
        fgMask = self.backSub.apply(frame, learningRate=lr)
        fgMask = self.filter_fg(fgMask)
        
        bboxes = self.bbox_extractor(fgMask)
        return bboxes

    __call__ = apply

def build_detector(var_thresh, min_size, close_kernel=5, open_kernel=2):
    # Define the Background Subtractor function
    backSub = build_bg_MOG2(var_thresh)
    # Define the Noise Filter
    filter_fg_func = MorphologyNoiseFilter(close_kernel=close_kernel, open_kernel=open_kernel)
    # Define the bboxes Finder
    bbox_extractor = lambda fgMask : oriented_bboxes_from_conected_components(fgMask, min_size=min_size)
    # Ensamble the Detector
    detector_model = ForegroundMaskObjectDetector(backSub, bbox_extractor, filter_fg_func)

    return detector_model
