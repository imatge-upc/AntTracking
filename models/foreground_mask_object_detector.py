
import numpy as np


def do_nothing(*args, **kargs):
    return *args, kargs

class ForegroundMaskObjectDetector():

    def __init__(self, backSub_func, bbox_extractor, filter_fg_func=None, train_frames=500, write_images_funct=False):
        self.backSub = backSub_func
        self.bbox_extractor = bbox_extractor
        self.filter_fg = filter_fg_func or do_nothing

        self.train_frames = train_frames
        self.start_write = train_frames

        self.write_images = write_images_funct
        self.out_ima = None # For debuging

    def reset(self):
        self.train_frames = self.start_write
        self.out_ima = None # For debuging


    def apply(self, frame):
        fgMask = self.backSub(frame)

        if self.train_frames > 0:
            self.train_frames = self.train_frames - 1
            return None
        
        fgMask = self.filter_fg(fgMask)

        if self.write_images:
            self.debug(frame, fgMask)
        
        bboxes = self.bbox_extractor(fgMask)
        return bboxes

    __call__ = apply


    def debug(self, frame, fgMask):
        h, w, d = frame.shape
        if self.out_ima is None:
            self.out_ima = np.zeros((h, 2 * w, d), dtype=np.uint8)

        self.out_ima[:, :w, :] = frame
        self.out_ima[:, w:, 0] = fgMask
        self.out_ima[:, w:, 1] = fgMask
        self.out_ima[:, w:, 2] = fgMask

        self.write_images(self.out_ima)
