
import cv2
import numpy as np
import pandas as pd


class ColorCropLoader():

    def __init__(self, ann_path, data_path, labels=True):
        self.ann_path = pd.read_csv(ann_path, delimiter=',', header=None)
        self.ann_path[2] = self.ann_path[0]
        self.ann_path[0] = f'{data_path}/' + self.ann_path[0]
        self.labels = labels

    def __getitem__(self, key):
        img = cv2.imread(self.ann_path.iat[key, 0], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.int32)

        if self.labels:
            label = self.ann_path.iat[key, 1]
            return img, label
        path = self.ann_path.iat[key, 2]
        return img, path

    def __len__(self):
        return len(self.ann_path)
    