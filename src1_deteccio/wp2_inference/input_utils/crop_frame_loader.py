# NOTE: SAHI does not use this because SAHI itself crops and reconstructs each image (one at a time)

from functools import lru_cache
import numpy as np

from ceab_ants.detection.utils.sliding_windows import sliceFrame


@lru_cache(maxsize=1)
def compute_overlap_mask(height, width, imgsz, overlap):
    overlap_mask = np.zeros((height, width), dtype=bool)
    stride = int(imgsz * (1 - overlap))
    for y_offset in range(0, height - imgsz, stride):
        overlap_mask[y_offset + imgsz - stride:y_offset + imgsz, :] = True
    for x_offset in range(0, width - imgsz, stride):
        overlap_mask[:, x_offset + imgsz - stride:x_offset + imgsz] = True
    return overlap_mask

class CropFrameLoader():
    
    def __init__(self, video_id=0, initial_frame=1, imgsz=640, overlap=0.2, th_color=np.inf):
        self.video_id = video_id
        self.fr = initial_frame - 1 
        self.imgsz = imgsz
        self.overlap = overlap
        self.th_color = th_color # TODO: This is part to a second model to filter out empty crops when emoty, it should become a callable input to maximize flexibility

    def preprocess(self, img, fr=None):
        # Once the frame is loaded, it is cropped keeping offsets and overlaps. Some crops may be skipped if a faster model can remove empty crops.

        self.fr = fr or self.fr + 1

        height, width = img.shape[:2]
        rgb_img = img[..., ::-1] if len(img.shape) == 3 else img

        crops, offsets = sliceFrame(rgb_img, self.imgsz, self.overlap, batch=True)
        overlap_mask = compute_overlap_mask(height, width, self.imgsz, self.overlap)

        model_input, offsets = zip(*[([crop], offset) for offset, crop in zip(offsets, crops) if np.any(crop < self.th_color)])
        model_input = list(model_input)
        offsets = list(offsets)
        
        num_crops = len(model_input) # num_inputs
        output_metadata = (self.video_id, self.fr, num_crops, self.imgsz, offsets, overlap_mask, height, width)

        return model_input, output_metadata
