
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


def main(pred_path, gt_path, colors=None):

    pred = pd.read_csv(pred_path, delimiter=',')
    gt = pd.read_csv(gt_path, delimiter=',', header=None, names=['path', 'gt_color'])
    
    colors = pred.columns[1:].tolist()
    
    pred.set_index('path', inplace=True)
    gt.set_index('path', inplace=True)

    # gt_color may not be on colors and colors value may be NaN if no model was applied
    df = pred.join(gt)

    # TODO: finish evaluation and save data on folder related to pred_path
