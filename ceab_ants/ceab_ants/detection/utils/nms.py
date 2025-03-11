
from functools import lru_cache
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from skimage.draw import polygon as sk_polygon


class BBox():

    def __init__(self, left, top, width, height, score=0):
        self.score = score

        self.left = left
        self.top = top
        self.width = width
        self.height = height
        
        self.right = left + width
        self.bottom = top + height

        self.x = left + width / 2
        self.y = top + height / 2

        self.area = height * width
    
    def l2_distance(self, other):
        return np.linalg.norm(np.asarray([self.x - other.x, self.y - other.y]))

    def intersection(self, other):
        width = max(0, min(self.right, other.right) - max(self.left, other.left))
        height = max(0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return BBox(max(self.left, other.left), max(self.top, other.top), width, height)
    
    def union_area(self, other):
        return self.area + other.area - self.intersection(other).area

    def iou(self, other):
        intersection = self.intersection(other).area
        if (self.area + other.area - intersection) == 0:
            return 0
        return intersection / (self.area + other.area - intersection) # self.intersection(other).area / self.union_area(other)
    
    def in_mask(self, mask):
        if self.left < 0 or self.right > mask.shape[1] - 1 or self.top < 0 or self.bottom > mask.shape[0] - 1:
            return True
        return np.any(mask[int(self.top):int(self.bottom) + 1, int(self.left):int(self.right) + 1])

class OBBox():

    def __init__(self, x, y, w, h, angle):
        rectangle = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
        rotated_rect = rotate(rectangle, angle, use_radians=True)
        
        self.obbox = translate(rotated_rect, x, y)
        self.x = x
        self.y = y

        self.area = w * h
    
    def l2_distance(self, other):
        return np.linalg.norm(np.asarray([self.x - other.x, self.y - other.y]))
    
    def intersection(self, other):
        return self.obbox.intersection(other.obbox) # it has the .area atribute
    
    def union_area(self, other):
        return self.area + other.area - self.intersection(other).area
    
    def iou(self, other):
        intersection = self.intersection(other).area
        if (self.area + other.area - intersection) == 0:
            return 0
        return intersection / (self.area + other.area - intersection) # self.intersection(other).area / self.union_area(other)
    
    def in_mask(self, mask):
        coords = np.array(self.obbox.exterior.coords)
        rows, cols = coords[:, 1].astype(int), coords[:, 0].astype(int)
        if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= mask.shape[0]) or np.any(cols >= mask.shape[1]):
            return True
        rr, cc = sk_polygon(rows, cols, shape=mask.shape)
        return np.any(mask[rr, cc])

def get_obbox(det):
    return det[[2, 3, 4, 5, 10]] # fr, id, x, y, w, h, conf, -1, -1, -1, angle

def bigAreaOneClassNMS(detections, th_iou=0.5, max_distance=50, get_bbox_funct=None, bbox_class=None):
    if get_bbox_funct is None:
        get_bbox_funct = lambda det : det[2:6] # returns left, top, width, height
    bbox_class = bbox_class or BBox
    
    cache_bbox = lru_cache(maxsize=len(detections))(bbox_class)

    sorted_detections = sorted(detections, key=lambda x : cache_bbox(*get_bbox_funct(x)).area, reverse=True)

    nms_detections = []
    nms_remove = []
    for i, det1 in enumerate(sorted_detections):
        if i in nms_remove:
            continue # It's already removed
        
        nms_detections.append(det1) # It is the biggest of its neighbours

        bbox1 = cache_bbox(*get_bbox_funct(det1))
        for j, det2 in enumerate(sorted_detections[(i + 1):]):
            bbox2 = cache_bbox(*get_bbox_funct(det2))

            if (bbox1.l2_distance(bbox2) < max_distance) and (bbox1.iou(bbox2) > th_iou):
                nms_remove.append(i + 1 + j) # It is a smaller neighbour
                
    return nms_detections

def bigAreaOneClassMaskedNMS(detections, mask, th_iou=0.5, max_distance=50, get_bbox_funct=None, bbox_class=None):
    if get_bbox_funct is None:
        get_bbox_funct = lambda det: det[2:6]  # Default to BBox: left, top, width, height
    bbox_class = bbox_class or BBox

    cache_bbox = lru_cache(maxsize=len(detections))(bbox_class)

    filtered_detections = []
    skipped_detections = []

    for det in detections:
        bbox = cache_bbox(*get_bbox_funct(det))
        if bbox.in_mask(mask):
            filtered_detections.append(det)
        else:
            skipped_detections.append(det)

    processed_nms = bigAreaOneClassNMS(filtered_detections, th_iou, max_distance, get_bbox_funct, bbox_class)

    return processed_nms + skipped_detections


def distanceNMS(detections, min_distance=35):
    # From FRAN's model
    # TODO: keep if the euclidean distance of the center with the kept centers is higher than min_distance
    pass
