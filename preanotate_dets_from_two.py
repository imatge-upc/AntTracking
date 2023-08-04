
from contextlib import contextmanager
import cv2 as cv
from distutils.util import strtobool
from docopt import docopt
import json
import numpy as np
import os
import pandas as pd
import sys


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

def valid_frames(validFramesFile):
    table = pd.read_csv(validFramesFile, sep='\t', names=['ini', 'fin', 'out'])
    table['out'] = table['out'].apply(json.loads)

    indices_ok = table[['ini', 'fin']].apply(lambda x : np.arange(x['ini'], x['fin'] + 1), axis=1)
    indices_ok = np.hstack(indices_ok)

    outs = table['out'][table['out'].apply(lambda x : x != [])].explode('out').to_frame()['out'].to_list()
    indices_nok = pd.DataFrame(outs, columns=['ini', 'fin']).apply(lambda x : np.arange(x['ini'], x['fin'] + 1), axis=1)
    indices_nok = np.hstack(indices_nok)

    indices_ok = indices_ok[~np.isin(indices_ok, indices_nok)]
    return indices_ok

def compute_iou(bbox1, bbox2):
    
    bbox1 = np.expand_dims(bbox1, 1) # N, 1, 5
    bbox2 = np.expand_dims(bbox2, 0) # 1, M, 5

    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 0] + bbox1[..., 2], bbox2[..., 0] + bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 1] + bbox1[..., 3], bbox2[..., 1] + bbox2[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    intersection_area_matrix = w * h

    area_test = bbox1[..., 2] * bbox1[..., 3]
    area_gt = bbox2[..., 2] * bbox2[..., 3]

    union_area_matrix = (area_test + area_gt - intersection_area_matrix)

    iou = intersection_area_matrix / union_area_matrix

    return iou #N, M

def nms_priority(prioDets, auxDets):
    prio_df = pd.DataFrame([prioDets[:, 0], prioDets[:, 2:6].tolist()], columns=['frame', 'bboxes'])
    aux_df = pd.DataFrame([auxDets[:, 0], auxDets[:, 2:6].tolist()], columns=['frame', 'bboxes'])

    prio_serie = prio_df.groupby(by='frame').apply(lambda x : np.vstack(x['bboxes']))
    prio_serie.name = 'prio'
    aux_serie = aux_df.groupby(by='frame').apply(lambda x : np.vstack(x['bboxes']))
    aux_serie.name = 'aux'

    nms_df = pd.concat([prio_serie, aux_serie], axis=1) #.fillna("")
    nms_df['IoU'] = nms_df.apply(lambda x : compute_iou(x['prio'], x['aux']) if x.notnull().all() else np.empty((0, 0)), axis=1)

    nms_df['FN'] = nms_df.apply(lambda x : (x['IoU'] == 0).all(axis=0).any(), axis=1)
    nms_df['FP'] = nms_df.apply(lambda x : (x['IoU'] == 0).all(axis=1).any(), axis=1)
    nms_df['TP'] = ~nms_df['FN'] & ~nms_df['FP']

    maybe_fn = nms_df.index.values[nms_df['FN']]
    maybe_fp = nms_df.index.values[nms_df['FP']]
    maybe_tp = nms_df.index.values[nms_df['TP']]

    return maybe_tp, maybe_fn, maybe_fp

def empty_where_there_should_be(prioDets, indices_ok):
    true_fn = indices_ok[~np.isin(indices_ok, prioDets[:, 0])]
    return true_fn

def more_than_one_object(prioDets):
    true_fp = np.array([fr for fr, count in zip(*np.unique(prioDets[:, 0], return_counts=True)) if count > 1])
    return true_fp

DOCTEXT = f"""
Usage:
  preanotate_dets_from_two.py <videoFile> <prioDetFile> <auxDetFile> <outPath> [<validFramesFile>] [--onePerFrame=<opf>] [--sampling_rate=<sr>]

Options:
  --onePerFrame=<opf>    If true, the assumption that not more than one object is in any frame will be used [default: True].
  --sampling_rate=<sr>      Number of frames skipped between saved images. [default: 1]

"""

if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    
    videoFile = args['<videoFile>']
    prioDetFile = args['<prioDetFile>']
    auxDetFile = args['<auxDetFile>']
    outPath = args['<outPath>']
    validFramesFile = args['<validFramesFile>'] # or None

    onePerFrame = strtobool(args['--onePerFrame'])
    sampling_rate = int(args['--sampling_rate'])

    maybe_tp_imgs_path = os.path.join(outPath, 'maybe_tp', 'images')
    maybe_tp_labels_path = os.path.join(outPath, 'maybe_tp', 'labels')
    maybe_tp_crops_path = os.path.join(outPath, 'maybe_tp', 'crops')

    maybe_fn_imgs_path = os.path.join(outPath, 'maybe_fn', 'images')
    maybe_fn_labels_path = os.path.join(outPath, 'maybe_fn', 'labels') #NOTE: created but empty

    maybe_fp_imgs_path = os.path.join(outPath, 'maybe_fp', 'images')
    maybe_fp_labels_path = os.path.join(outPath, 'maybe_fp', 'labels')
    maybe_fp_crops_path = os.path.join(outPath, 'maybe_fp', 'crops')

    prioDets = np.loadtxt(prioDetFile, delimiter=',')
    auxDets = np.loadtxt(auxDetFile, delimiter=',')

    maybe_tp, maybe_fn, maybe_fp = nms_priority(prioDets, auxDets)

    if validFramesFile:
        indices_ok = valid_frames(validFramesFile)
        true_fn = empty_where_there_should_be(prioDets, indices_ok)
    else:
        indices_ok = np.sort(np.unique(np.concatenate(prioDets[:, 0], auxDets[:, 0])))
        true_fn = np.empty((0))

    if onePerFrame:
        true_fp = more_than_one_object(prioDets)
    else:
        true_fp = np.empty((0))

    maybe_tp = maybe_tp[~np.isin(maybe_tp, true_fp)]

    os.makedirs(outPath, exist_ok=False)
    os.makedirs(maybe_tp_imgs_path)
    os.makedirs(maybe_tp_labels_path)
    os.makedirs(maybe_tp_crops_path)
    os.makedirs(maybe_fn_imgs_path)
    os.makedirs(maybe_fn_labels_path)
    os.makedirs(maybe_fp_imgs_path)
    os.makedirs(maybe_fp_labels_path)
    os.makedirs(maybe_fp_crops_path)

    with VideoCapture(videoFile) as capture:
        
        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

        mot2yolo = lambda trk : ['0', f'{(trk[2] + (trk[4] / 2)) / width}', f'{(trk[3] + (trk[5] / 2)) / height}', f'{trk[4] / width}', f'{trk[5] / height}']

        for frame_id in indices_ok[::sampling_rate]:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_id - 1)

            _, frame = capture.read()
            if frame is None:
                print (f'Frame {frame_id} is None', file=sys.stderr)
                break
            
            filename = f'{frame_id:06}.png'
            labels_filename = f'{frame_id:06}.txt'
            
            if frame_id in maybe_tp:
                bboxes = prioDets[prioDets[:, 0] == frame_id, :]
                labels = '\n'.join([' '.join(mot2yolo(trk)) for trk in bboxes])

                for i, bbox in enumerate(bboxes):
                    crop_filename = f'{frame_id:06}_{i:02}.txt'

                    crop = frame[:, bbox[3] : bbox[3] + bbox[5], bbox[2] : bbox[2] + bbox[4]].numpy()
                    crop = np.moveaxis(crop, [0, 1, 2], [2, 0, 1])

                    cv.imwrite(os.path.join(maybe_tp_crops_path, crop_filename), crop)
                
                cv.imwrite(os.path.join(maybe_tp_imgs_path, filename), frame)
                with open(os.path.join(maybe_tp_labels_path, labels_filename), 'w') as f:
                    f.write(labels)

            elif frame_id in maybe_fn or frame_id in true_fn:
                cv.imwrite(os.path.join(maybe_fn_imgs_path, filename), frame)
            
            elif frame_id in maybe_fp or frame_id in true_fp:
                bboxes = prioDets[prioDets[:, 0] == frame_id, :]
                labels = '\n'.join([' '.join(mot2yolo(trk)) for trk in bboxes])

                for i, bbox in enumerate(bboxes):
                    crop_filename = f'{frame_id:06}_{i:02}.txt'

                    crop = frame[:, bbox[3] : bbox[3] + bbox[5], bbox[2] : bbox[2] + bbox[4]].numpy()
                    crop = np.moveaxis(crop, [0, 1, 2], [2, 0, 1])

                    cv.imwrite(os.path.join(maybe_fp_crops_path, crop_filename), crop)
                
                cv.imwrite(os.path.join(maybe_fp_imgs_path, filename), frame)
                with open(os.path.join(maybe_fp_labels_path, labels_filename), 'w') as f:
                    f.write(labels)
  