import numpy as np
import pandas as pd
from typing import List
import cv2
import sys


def bbox_center (bbox:np.ndarray) -> np.ndarray:
    '''
    bbox in format x1y1wh
    '''
    return bbox[0:2] + bbox[2:4]/2



def track_velocities(track_bboxes:List[np.ndarray]) -> Tuple[List[float]]:

    vel_modul = list()
    vel_phase = list()
    for ii in range(1, len(track_bboxes)):
        prev = bbox_center(track_bboxes[ii-1])
        curr = bbox_center(track_bboxes[ii])
        disp = curr - prev

        vel_modul.append(np.linalg.norm(disp))
        vel_phase.append(np.arctan2(disp[1], disp[0]))

    return vel_modul, vel_phase
            


        

# ChatGPT generated code
def fill_lists(list1, list2):
    i, j = 0, 0
    len1, len2 = len(list1), len(list2)
    result1, result2 = [], []
    while i < len1 or j < len2:
        if i < len1 and j < len2:
            if list1[i][0] < list2[j][0]:
                result1.append(list1[i])
                result2.append([list1[i][0], -1, -1, -1, -1, -1, -1])
                i += 1
            elif list1[i][0] > list2[j][0]:
                result1.append([list2[j][0], -1, -1, -1, -1, -1, -1])
                result2.append(list2[j])
                j += 1
            else:
                result1.append(list1[i])
                result2.append(list2[j])
                i += 1
                j += 1
        elif i < len1:
            result1.append(list1[i])
            result2.append([list1[i][0], -1, -1, -1, -1, -1, -1])
            i += 1
        elif j < len2:
            result1.append([list2[j][0], -1, -1, -1, -1, -1, -1])
            result2.append(list2[j])
            j += 1
    return result1, result2



# Aligned data frames
#def track_alignment (track1:pd.DataFrame, track:pd.DataFrame) -> tuple(pd.DataFrame,pd.DataFrame):

# ChatGPT generated code, corrected by JRMR
def find_overlap(list1, list2):
    overlap_indices = []
    for ii in range(len(list1)):
        rect1 = list1[ii]
        rect2 = list2[ii]
        x1, y1, w1, h1 = rect1[0], rect1[1], rect1[2], rect1[3]
        x2, y2, w2, h2 = rect2[0], rect2[1], rect2[2], rect2[3]

        if (x1 < x2 + w2) and (x1 + w1 > x2) and (y1 < y2 + h2) and (y1 + h1 > y2):
            overlap_indices.append(ii)

    return overlap_indices


# ------------------------------------------------------------


def crop_rectangles (ima:np.ndarray, rectangles:List[int], size:int = -1)->List[np.ndarray]:
    '''
    rectangles: list of rectangles in x1y1wh format
    '''
    kernel3 = np.ones((3,3),np.uint8)
    
    out    = list()
    for rect in rectangles:
        x,y,w,h = rect
        cx,cy = map(int,map(round,(x+w/2,y+h/2)))
        if size == -1:
            size = max(w,h)
        x1,y1 = cx-size//2, cy-size//2
        
        crop = ima[y1:y1+size,x1:x1+size,:]
        print (crop.shape)
        crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel3)
                
        out.append(crop)
        
    return out
    

def binarize_crops(crops:List[np.ndarray])->List[np.ndarray]:
    out = list()
    for crop in crops:
        ret,th = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)        
        out.append(th)

    return out


def save_crops(crops:List[np.ndarray], frame_id:int, folder:str, suffix:str = ''):
    for crop in crops:
        cv2.imwrite(f'{folder}/{frame_id:06d}{suffix}.png', crop)


def get_rectangles_from_frame(df:pd.DataFrame, fr:int)->List[List[int]]:
    return df.loc[df['frame']==fr][['x','y','w','h']].to_numpy().astype(int).tolist()


def find_orientations(crops:List[np.ndarray]) -> List[float]:
    '''
    Find the orientation of each object (binary blob) in a list.
    Angle is CW/CCW? starting at ???
    '''
    orientations = list()
    for crop in crops:
        ant_coords = np.transpose(np.nonzero(cv2.bitwise_not(ima)))
        orientations.append(cv2.fitEllipse(ant_coords)[2])
    return orientations


def find_orientations_pca(crops:List[np.ndarray]) -> List[float]:
    '''
    Find the orientation of each object (binary blob) in a list.
    Angle is CW/CCW? starting at ???
    https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    '''
    orientations = list()
    for crop in crops:
        y, x = np.nonzero(crops)
        x = x - np.mean(x)
        y = y - np.mean(y)
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        #x_v2, y_v2 = evecs[:, sort_indices[1]]
        
        orientations.append(np.arctan2(y_v1, x_v1))

    return orientations



def process_track(df:pd.DataFrame,input_video:str, start_frame:int = 1, stop_frame:int = -1, out_dir:str = '.'):

    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(input_video))
    if not capture.isOpened():
        print('Unable to open: ' + input_video, file=sys.stderr)
        exit(0)

    fr = start_frame

    # https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)

    while True:
        if stop_frame > 0 and fr > stop_frame:
            break
            
        if fr%100 == 0:
            print (f'Processing frame {fr}', file=sys.stderr)
        ret, frame = capture.read()
        if frame is None:
            break

        rectangles = get_rectangles_from_frame(df, fr)
        crops = crop_rectangles(frame, rectangles)
        bw_crops = binarize_crops(crops)
        save_crops(crops, fr, out_dir, '_ori')
        save_crops(bw_crops, fr, out_dir, '_bw')

        fr += 1
        
    # Release the video capture and writer objects
    capture.release()
