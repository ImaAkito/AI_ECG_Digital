import numpy as np
import torch
from typing import Optional

def multilabel_accuracy(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    correct = (y_true == y_pred_bin).all(axis=1).sum()
    return correct / y_true.shape[0]

def multilabel_f1(y_true, y_pred, threshold=0.5, average='macro'):
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    tp = (y_true * y_pred_bin).sum(axis=0)
    fp = ((1 - y_true) * y_pred_bin).sum(axis=0)
    fn = (y_true * (1 - y_pred_bin)).sum(axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    if average == 'macro':
        return np.nanmean(f1)
    elif average == 'micro':
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)
    else:
        return f1

def multilabel_coverage(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    covered = ((y_true * y_pred_bin).sum(axis=1) > 0).sum()
    return covered / y_true.shape[0]

def multilabel_precision(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    tp = (y_true * y_pred_bin).sum(axis=0)
    fp = ((1 - y_true) * y_pred_bin).sum(axis=0)
    precision = tp / (tp + fp + 1e-8)
    return np.nanmean(precision)

def multilabel_recall(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    tp = (y_true * y_pred_bin).sum(axis=0)
    fn = (y_true * (1 - y_pred_bin)).sum(axis=0)
    recall = tp / (tp + fn + 1e-8)
    return np.nanmean(recall)

def multilabel_auroc(y_true, y_pred): #AUROC for multilabel (without sklearn)
    n_classes = y_true.shape[1]
    aucs = []
    for i in range(n_classes):
        t = y_true[:, i]
        p = y_pred[:, i]
        # Sorting by probability 
        desc = np.argsort(-p)
        t = t[desc]
        p = p[desc]
        tp = 0
        fp = 0
        tp_prev = 0
        fp_prev = 0
        auc = 0.0
        n_pos = t.sum()
        n_neg = len(t) - n_pos
        for j in range(len(t)):
            if t[j]:
                tp += 1
            else:
                fp += 1
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
            tp_prev = tp
            fp_prev = fp
        if n_pos * n_neg > 0:
            auc = auc / (n_pos * n_neg)
            aucs.append(auc)
    return np.nanmean(aucs) if aucs else float('nan') 