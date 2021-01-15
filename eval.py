import numpy as np
import glob
from pathlib import Path
from misc import read_boxes


def precision_recall(preds, iou, n_gt):
    ious = np.array([p[0] for p in preds])
    tp = np.cumsum(ious >= iou)
    fn = n_gt - tp
    fp = np.cumsum(ious < iou)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return prec, rec


def count_gt(subset):
    total = 0
    for filename in subset:
        total += len(read_boxes(filename))
    return total