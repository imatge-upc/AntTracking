
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


COLORS = ['black', 'green', 'red', 'pink'] # etc


def main(pred_path, gt_path, colors=None):
    colors = colors or COLORS

    pred = pd.read_csv(pred_path, delimiter=',', header=None, names=['file', 'pred_color'])
    gt = pd.read_csv(gt_path, delimiter=',', header=None, names=['file', 'gt_color'])

    pred.set_index('file', inplace=True)
    gt.set_index('file', inplace=True)

    df = pred.join(gt).dropna()

    total = len(df)
    confusion = confusion_matrix(df.gt_color, df.pred_color, labels=colors)

    report = classification_report(df.gt_color, df.pred_color, digits=4)

    # TODO: finish evaluation and save data on folder related to pred_path
