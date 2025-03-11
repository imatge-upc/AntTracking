
import numpy as np
from scipy.optimize import linear_sum_assignment

from ceab_ants.bbox_metrics.bbox_metrics import iou_bbox_batch


def iou_th_assigment(pred, gt, th):
    iou_matrix = iou_bbox_batch(pred, gt) > th
    return iou_matrix

def average_precision(pred, gt, get_pred=None, get_scores=None, get_gt=None, assignment_funct=None, use_hungarian=False, additional_metrics=False):
    # pred is the predictions of the model given a setting (by instance, confidence score higher than 0.5)

    if len(pred) == 0 and len(gt) == 0:
        return None

    get_pred = get_pred or (lambda x : x[:, :-1])
    get_scores = get_scores or (lambda x : x[:, -1])
    get_gt = get_gt or (lambda x : x)

    assignment_funct = assignment_funct or (lambda pred, gt : iou_th_assigment(pred, gt, th=0.5)) # input: pred, gt

    scores = get_scores(pred)
    pred = get_pred(pred)
    gt = get_gt(gt)

    pred = pred[np.argsort(scores)[::-1], ...]
    assigment_matrix = assignment_funct(pred, gt) # N rows (preds) x M columns (gt)

    if use_hungarian and assigment_matrix.size > 0: # keep the PREDs order but sorts GTs to get optimal results as GT order is not relevant
        row_ind, col_ind = linear_sum_assignment(-assigment_matrix.astype(float))
        sorted_indices = np.argsort(row_ind)
        col_ind = col_ind[sorted_indices]
        
        matched_gt = gt[col_ind]
        unmatched_gt_mask = np.ones(len(gt), dtype=bool)
        unmatched_gt_mask[col_ind] = False
        unmatched_gt = gt[unmatched_gt_mask]
        gt = np.concatenate([matched_gt, unmatched_gt], axis=0)
        assigment_matrix = np.hstack([assigment_matrix[:, col_ind], np.zeros((assigment_matrix.shape[0], len(unmatched_gt)), dtype=bool)])

    tp_mask = np.zeros_like(assigment_matrix, dtype=bool)
    used_predictions = np.zeros(assigment_matrix.shape[0], dtype=bool)
    for j in range(assigment_matrix.shape[1]):
        match_indices = np.where(assigment_matrix[:, j])[0]
        match_indices = match_indices[~used_predictions[match_indices]]
        if match_indices.size > 0:
            tp_mask[match_indices[0], j] = True
            used_predictions[match_indices[0]] = True

    # How many were filtered correctly by the assignment (given a TP -> number of False in column)
    tn_given_a_tp = np.sum(~assigment_matrix[:, np.any(tp_mask, axis=0)])
    # How many were filtered correctly by another TP
    fp_discarded = np.sum(assigment_matrix & ~tp_mask)

    assigment_matrix &= tp_mask
    has_match = np.any(assigment_matrix, axis=1)

    # TP for any True in gt (column) for a row
    v_tp = has_match
    # FP when all gt (check all columns) are False for a given pred (row)
    v_fp = ~has_match
    # FN when all pred (check all rows) are False for a given gt (column) <- after reassign because a TP was already found for a column (double possitives are FP)
    v_fn = ~np.any(assigment_matrix, axis=0)

    if len(has_match) != 0:
        tp_cumsum = np.cumsum(v_tp).astype(float)
        fp_cumsum = np.cumsum(v_fp).astype(float)
    else:
        tp_cumsum = np.array([0], dtype=float)
        fp_cumsum = np.array([0], dtype=float)

    # Empty gt with empty pred is considered perfect, please, in case of "empty gt and empty pred" omit the frame in a higher level to get more relevant information
    recall = tp_cumsum / (tp_cumsum[-1] + np.sum(v_fn)) if (tp_cumsum[-1] + np.sum(v_fn)) > 0 else np.ones_like(tp_cumsum, dtype=tp_cumsum.dtype)
    precision = np.divide(tp_cumsum, (tp_cumsum + fp_cumsum), out=np.ones_like(tp_cumsum, dtype=tp_cumsum.dtype), where=(tp_cumsum + fp_cumsum) > 0)
    
    # From ultralytics https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L505
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([precision[0]], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    #solution 1: x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    #solution 1: ap = np.trapz(np.interp(x, mrec, mpre), x)
    #solution 2: i = np.where(mrec[1:] != mrec[:-1])[0]
    #solution 2: ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    # Mine: Reflects the Impact of FP and with larger datasets at once, the precision-recall curve will become more stable.
    # Mine: However, in object detection, it needs a lot of objects per frame or more complex input like "mosaic" of frames
    mrec_midpoints = (mrec[1:] + mrec[:-1]) / 2
    mrec_all = np.sort(np.concatenate((mrec, mrec_midpoints)))
    mpre_interp = np.interp(mrec_all, mrec, mpre)
    ap = np.trapz(mpre_interp, mrec_all)
    
    if not additional_metrics:
        return ap
    else:

        fn = np.sum(v_fn)
        tp = np.sum(v_tp)
        fp = np.sum(v_fp)

        metrics = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision[-1],
            'recall': recall[-1],
            "tn_given_a_tp" : tn_given_a_tp,
            "fp_discarded" : fp_discarded
        }

        return ap, metrics

def mean_average_precision(settings, get_pred=None, get_scores=None, get_gt=None, use_hungarian=False, additional_metrics=False):
    # the mean of average_precision given a set of { (pred, gt, assignment_funct), ... }
    
    ap_list = []
    metrics_list = []
    for pred, gt, assignment_funct in settings:
        ap = average_precision(pred, gt, get_pred, get_scores, get_gt, assignment_funct, use_hungarian=use_hungarian, additional_metrics=additional_metrics)
        if additional_metrics:
            ap, metrics = ap
            metrics_list.append(metrics)

        if ap is not None:
            ap_list.append(ap)
    
    mAP = np.mean(ap_list) if ap_list else 0
    if not additional_metrics:
        return mAP
    else:
        return mAP, metrics_list


def mAP_results(pred, gt, assignment_funct_list, get_pred=None, get_scores=None, get_gt=None, use_hungarian=False, additional_metrics=False):
    # Usually the mAP a single set of precomputed results

    settings = [(pred, gt, assignment_funct) for assignment_funct in assignment_funct_list]
    mAP = mean_average_precision(settings, get_pred, get_scores, get_gt, use_hungarian, additional_metrics)
    
    if additional_metrics:
        mAP, metrics_list = mAP
        return mAP, metrics_list
    return mAP

def mAP_iou(pred, gt, thresholds=None, get_pred=None, get_scores=None, get_gt=None, iou_funct=None, use_hungarian=False, additional_metrics=False):
    # In single class object detection, usually the assigment is based on iou thereshold

    iou_funct = iou_funct or iou_bbox_batch
    thresholds = thresholds if thresholds is not None else [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    settings = [(pred, gt, ( lambda pred, gt, th=th: iou_funct(pred, gt) > th )) for th in thresholds]
    mAP = mean_average_precision(settings, get_pred, get_scores, get_gt, use_hungarian, additional_metrics)

    if additional_metrics:
        mAP, metrics_list = mAP
        return mAP, metrics_list
    return mAP
