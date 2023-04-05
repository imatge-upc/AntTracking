
from contextlib import contextmanager
import cv2 as cv
import sys
from torchvision.models import vit_b_32 as VisionTransformer

from docopts.help_deepsort_detection import parse_args
from models.deepsort_utils.feature_extractor import FeatureExtractor
from models.deepsort_utils.apparence_bbox_detector import ApparenceBBoxDetector
from models.foreground_mask_object_detector import ForegroundMaskObjectDetector


@contextmanager
def VideoCapture(input_video):

    # findFileOrKeep allows more searching paths
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(input_video))
    if not capture.isOpened():
        print('Unable to open: ' + input_video, file=sys.stderr)
        exit(0)

    try:
        yield capture
    finally:
        # Release the video capture object at the end
        capture.release()


class MorphologyNoiseFilter():
    # Remove noise  https://stackoverflow.com/questions/30369031/remove-spurious-small-islands-of-noise-in-an-image-python-opencv

    def __init__(self):
        self.se1 = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
        self.se2 = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    
    def __call__(self, fgMask):
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, self.se1)
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN,  self.se2)
        return fgMask

def bboxes_from_conected_components(fgMask, min_size=20, connectivity=4, ltype=cv.CV_32S):
    analysis = cv.connectedComponentsWithStats(fgMask, connectivity, ltype)
    (totalLabels, labels, stats, centroids) = analysis

    bboxes = []
    for ii in range(1, totalLabels):
        # extract the connected component statistics for the current label
        x = stats[ii, cv.CC_STAT_LEFT]
        y = stats[ii, cv.CC_STAT_TOP]
        w = stats[ii, cv.CC_STAT_WIDTH]
        h = stats[ii, cv.CC_STAT_HEIGHT]
        
        if w > min_size and h > min_size:
            bboxes.append((x, y, w, h))
    
    return bboxes

if __name__ == '__main__':
    # read arguments
    input_video, detection_file, subs_alg,\
    var_thresh, filter_fg, write_images,\
    out_dir, min_size, start_write, stop_frame = parse_args(sys.argv)


    # Define the Background Subtractor function
    if subs_alg == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2(varThreshold=var_thresh, detectShadows=False)
    else:
        backSub = cv.createBackgroundSubtractorKNN()
    backSub_func = lambda frame : backSub.apply(frame)

    # Define the Noise Filter
    if filter_fg:
        filter_fg_func = MorphologyNoiseFilter()
    else:
        filter_fg_func = None

    # Define the bboxes Finder
    bbox_extractor = lambda fgMask : bboxes_from_conected_components(fgMask, min_size=min_size)

    # Define debug function
    if write_images:
        def write_images(out_ima):
            cv.imwrite (f'{out_dir}/out_{fr:06d}.png', out_ima)


    # Ensamble the Detector
    detector_model = ForegroundMaskObjectDetector(backSub_func, bbox_extractor, filter_fg_func, start_write, write_images)
    
    apparence_model = FeatureExtractor(VisionTransformer(weights='DEFAULT'), ['encoder']) # Output list of 1 Tensor [#bboxes, 50, 768]
    apparence_model.eval()
    #apparence_model_applier = lambda x : apparence_model(x)[0] # Output [#bboxes, 50, 768]
    apparence_model_applier = lambda x : apparence_model(x)[0].mean(2).numpy(force=True) # Output [#bboxes, 50]

    model = ApparenceBBoxDetector(detector_model, apparence_model_applier)


    # Apply the model
    fr = 0
    with VideoCapture(input_video) as capture:
        with open(detection_file, 'w') as out_file:
            while stop_frame <= 0 or fr <= stop_frame:
                fr = fr + 1
                
                if fr % (start_write or 1) == 0:
                    print (f'Processing frame {fr}', file=sys.stderr)

                _, frame = capture.read()
                if frame is None:
                    print (f'Frame {fr} is None', file=sys.stderr)
                    break
                    
                bboxes = model(frame)
                if bboxes is None:
                    continue # Training Background
                
                if len(bboxes) > 0:
                    # bbox = (x1, y1, w, h)
                    MOTDet_line = lambda fr, bbox : f'{fr}, -1, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, 1, -1, -1, -1, {", ".join([str(b) for b in bbox[4:]])}'[:-2]
                    detection_text = '\n'.join([MOTDet_line(fr, bbox) for bbox in bboxes])
                    print (detection_text, file=out_file)
