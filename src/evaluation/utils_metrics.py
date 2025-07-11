"""Utilities for computing metrics.

NOTE that only the widely used metrics are implemented here,
challenge (e.g. CinC, CPSC series) specific metrics are not included.

"""

import warnings
from numbers import Real
from typing import Dict, Optional, Sequence, Tuple, Union

import einops
import numpy as np
import torch
from torch import Tensor

from ..cfg import DEFAULTS
from .misc import add_docstring
from .utils_data import ECGWaveForm, ECGWaveFormNames, masks_to_waveforms

__all__ = [
    "top_n_accuracy",
    "confusion_matrix",
    "ovr_confusion_matrix",
    "QRS_score",
    "metrics_from_confusion_matrix",
    "compute_wave_delineation_metrics",
]


def top_n_accuracy(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    n: Union[int, Sequence[int]] = 1,
) -> Union[float, Dict[str, float]]:
    """Compute top n accuracy.

    Parameters
    ----------
    labels : numpy.ndarray or torch.Tensor
        Labels of class indices,
        of shape ``(batch_size,)`` or ``(batch_size, d_1, ..., d_m)``.
    outputs : numpy.ndarray or torch.Tensor
        Predicted probabilities, of shape ``(batch_size, num_classes)``
        or ``(batch_size, d_1, ..., d_m, num_classes)``
        or ``(batch_size, num_classes, d_1, ..., d_m)``.
    n : int or List[int]
        Top n to be considered.

    Returns
    -------
    acc : float or dict of float
        Top n accuracy.

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> labels, outputs = DEFAULTS.RNG_randint(0, 9, (100)), DEFAULTS.RNG.uniform(0, 1, (100, 10))  # 100 samples, 10 classes
    >>> top_n_accuracy(labels, outputs, 3)
    0.32
    >>> top_n_accuracy(labels, outputs, [1,3,5])
    {'top_1_acc': 0.12, 'top_3_acc': 0.32, 'top_5_acc': 0.52}

    """
    assert outputs.shape[0] == labels.shape[0], "outputs and labels must have the same batch size"
    labels, outputs = torch.as_tensor(labels), torch.as_tensor(outputs)
    batch_size, n_classes, *extra_dims = outputs.shape
    if isinstance(n, int):
        ln = [n]
    else:
        ln = n
    acc = {}
    for _n in ln:
        key = f"top_{_n}_acc"
        _, indices = torch.topk(outputs, _n, dim=1)  # of shape (batch_size, n) or (batch_size, n, d_1, ..., d_n)
        pattern = " ".join([f"d_{i+1}" for i in range(len(extra_dims))])
        pattern = f"batch_size {pattern} -> batch_size n {pattern}"
        correct = torch.sum(indices == einops.repeat(labels, pattern, n=_n))
        acc[key] = correct.item() / outputs.shape[0]
        for d in extra_dims:
            acc[key] = acc[key] / d
    if len(ln) == 1:
        return acc[key]
    return acc


def confusion_matrix(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute a binary confusion matrix

    The columns are ground truth labels and rows are predicted labels.

    Parameters
    ----------
    labels : numpy.ndarray or torch.Tensor
        Binary labels, of shape ``(n_samples, n_classes)``,
        or indices of each label class, of shape ``(n_samples,)``.
    outputs : numpy.ndarray or torch.Tensor
        Binary outputs, of shape ``(n_samples, n_classes)``,
        or indices of each class predicted, of shape ``(n_samples,)``.
    num_classes : int, optional
        Number of classes.
        If `labels` and `outputs` are both of shape ``(n_samples,)``,
        then `num_classes` must be specified.

    Returns
    -------
    cm : numpy.ndarray
        Confusion matrix, of shape ``(n_classes, n_classes)``.

    """
    labels, outputs = one_hot_pair(labels, outputs, num_classes)
    assert np.shape(labels) == np.shape(outputs), "labels and outputs must have the same shape"
    assert all([value in (0, 1) for value in np.unique(labels)]), "labels must be binary"
    assert all([value in (0, 1) for value in np.unique(outputs)]), "outputs must be binary"

    num_samples, num_classes = np.shape(labels)

    cm = np.zeros((num_classes, num_classes))
    for k in range(num_samples):
        i = np.argmax(outputs[k, :])
        j = np.argmax(labels[k, :])
        cm[i, j] += 1

    return cm


def one_vs_rest_confusion_matrix(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute binary one-vs-rest confusion matrices.

    Columns are ground truth labels and rows are predicted labels.

    Parameters
    ----------
    labels : numpy.ndarray or torch.Tensor
        Binary labels, of shape ``(n_samples, n_classes)``,
        or indices of each label class, of shape ``(n_samples,)``.
    outputs : numpy.ndarray or torch.Tensor
        Binary outputs, of shape ``(n_samples, n_classes)``,
        or indices of each class predicted, of shape ``(n_samples,)``.
    num_classes : int, optional
        number of classes.
        If `labels` and `outputs` are both of shape ``(n_samples,)``,
        then `num_classes` must be specified.

    Returns
    -------
    ovr_cm : numpy.ndarray
        One-vs-rest confusion matrix, of shape ``(n_classes, 2, 2)``.

    """
    labels, outputs = one_hot_pair(labels, outputs, num_classes)
    assert np.shape(labels) == np.shape(outputs), "labels and outputs must have the same shape"
    assert all([value in (0, 1) for value in np.unique(labels)]), "labels must be binary"
    assert all([value in (0, 1) for value in np.unique(outputs)]), "outputs must be binary"

    num_samples, num_classes = np.shape(labels)

    ovr_cm = np.zeros((num_classes, 2, 2))
    for i in range(num_samples):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                ovr_cm[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                ovr_cm[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                ovr_cm[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                ovr_cm[j, 1, 1] += 1

    return ovr_cm


# alias
ovr_confusion_matrix = one_vs_rest_confusion_matrix


_METRICS_FROM_CONFUSION_MATRIX_PARAMS = """
    Compute macro {metric}, and {metrics} for each class.

    Parameters
    ----------
    labels : numpy.ndarray or torch.Tensor
        Binary labels, of shape ``(n_samples, n_classes)``,
        or indices of each label class, of shape ``(n_samples,)``.
    outputs : numpy.ndarray or torch.Tensor
        Probability outputs, of shape ``(n_samples, n_classes)``,
        or binary outputs, of shape ``(n_samples, n_classes)``,
        or indices of each class predicted, of shape ``(n_samples,)``.
    num_classes : int, optional
        Number of classes.
        If `labels` and `outputs` are both of shape ``(n_samples,)``,
        then `num_classes` must be specified.
    weights : numpy.ndarray or torch.Tensor, optional
        Weights for each class, of shape ``(n_classes,)``,
        used to compute macro {metric}.
    thr : float, default: 0.5
        Threshold for binary classification,
        valid only if `outputs` is of shape ``(n_samples, n_classes)``.
    fillna : bool or float, default: 0.0
        If is False, then NaN will be left in the result.
        If is True, then NaN will be filled with 0.0.
        If is a float, then NaN will be filled with the specified value.
"""


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="metrics", metrics="metrics"),
    "prepend",
)
def metrics_from_confusion_matrix(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
    thr: float = 0.5,
    fillna: Union[bool, float] = 0.0,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Returns
    -------
    metrics : dict
        Metrics computed from the one-vs-rest confusion matrix.

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> # binary labels (100 samples, 10 classes, multi-label)
    >>> labels = DEFAULTS.RNG_randint(0, 1, (100, 10))
    >>> # probability outputs (100 samples, 10 classes, multi-label)
    >>> outputs = DEFAULTS.RNG.random((100, 10))
    >>> metrics = metrics_from_confusion_matrix(labels, outputs)
    >>> # binarize outputs (100 samples, 10 classes, multi-label)
    >>> outputs = DEFAULTS.RNG_randint(0, 1, (100, 10))
    >>> # would raise
    >>> # RuntimeWarning: `outputs` is probably binary or categorical, AUC may be incorrect
    >>> metrics = metrics_from_confusion_matrix(labels, outputs)
    >>> # categorical outputs (100 samples, 10 classes)
    >>> outputs = DEFAULTS.RNG_randint(0, 9, (100,))
    >>> # would raise
    >>> # RuntimeWarning: `outputs` is probably binary or categorical, AUC may be incorrect
    >>> metrics = metrics_from_confusion_matrix(labels, outputs)

    """
    outputs_ndim = np.ndim(outputs)
    labels, outputs = one_hot_pair(labels, outputs, num_classes)
    num_samples, num_classes = np.shape(labels)

    # probability outputs to binary outputs
    bin_outputs = np.zeros_like(outputs, dtype=int)
    bin_outputs[outputs >= thr] = 1
    bin_outputs[outputs < thr] = 0
    if np.unique(outputs).size == 2:
        warnings.warn("`outputs` is probably binary or categorical, AUC may be incorrect", RuntimeWarning)

    ovr_cm = ovr_confusion_matrix(labels, bin_outputs)

    # sens: sensitivity, recall, hit rate, or true positive rate
    # spec: specificity, selectivity or true negative rate
    # prec: precision or positive predictive value
    # npv: negative predictive value
    # jac: jaccard index, threat score, or critical success index
    # acc: accuracy
    # phi: phi coefficient, or matthews correlation coefficient
    # NOTE: never use repeat here, because it will cause bugs
    # sens, spec, prec, npv, jac, acc, phi = list(repeat(np.zeros(num_classes), 7))
    sens, spec, prec, npv, jac, acc, phi = [np.zeros(num_classes) for _ in range(7)]
    auroc = np.zeros(num_classes)  # area under the receiver-operater characteristic curve (ROC AUC)
    auprc = np.zeros(num_classes)  # area under the precision-recall curve
    for k in range(num_classes):
        tp, fp, fn, tn = (
            ovr_cm[k, 0, 0],
            ovr_cm[k, 0, 1],
            ovr_cm[k, 1, 0],
            ovr_cm[k, 1, 1],
        )
        if tp + fn > 0:
            sens[k] = tp / (tp + fn)
        else:
            sens[k] = float("nan")
        if tp + fp > 0:
            prec[k] = tp / (tp + fp)
        else:
            prec[k] = float("nan")
        if tn + fp > 0:
            spec[k] = tn / (tn + fp)
        else:
            spec[k] = float("nan")
        if tn + fn > 0:
            npv[k] = tn / (tn + fn)
        else:
            npv[k] = float("nan")
        if tp + fn + fp > 0:
            jac[k] = tp / (tp + fn + fp)
        else:
            jac[k] = float("nan")
        acc[k] = (tp + tn) / num_samples
        phi[k] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if outputs_ndim == 1:
            auroc[k] = np.nan
            auprc[k] = np.nan
            continue
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_samples and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr_ = np.zeros(num_thresholds)
        tnr_ = np.zeros(num_thresholds)
        ppv_ = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr_[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr_[j] = float("nan")
            if fp[j] + tn[j]:
                tnr_[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr_[j] = float("nan")
            if tp[j] + fp[j]:
                ppv_[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv_[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr_[j + 1] - tpr_[j]) * (tnr_[j + 1] + tnr_[j])
            auprc[k] += (tpr_[j + 1] - tpr_[j]) * ppv_[j + 1]

    fnr = 1 - sens  # false negative rate, miss rate
    fpr = 1 - spec  # false positive rate, fall-out
    fdr = 1 - prec  # false discovery rate
    for_ = 1 - npv  # false omission rate
    plr = sens / fpr  # positive likelihood ratio
    nlr = fnr / spec  # negative likelihood ratio
    pt = np.sqrt(fpr) / (np.sqrt(sens) + np.sqrt(fpr))  # prevalence threshold
    ba = (sens + spec) / 2  # balanced accuracy
    f1 = 2 * sens * prec / (sens + prec)  # f1-measure
    fm = np.sqrt(prec * sens)  # fowlkes-mallows index
    bm = sens + spec - 1  # informedness, bookmaker informedness
    mk = prec + npv - 1  # markedness
    dor = plr / nlr  # diagnostic odds ratio

    if weights is None:
        _weights = np.ones(num_classes)
    else:
        _weights = weights / np.mean(weights)
    metrics = {}
    for m in [
        "sens",  # sensitivity, recall, hit rate, or true positive rate
        "spec",  # specificity, selectivity or true negative rate
        "prec",  # precision or positive predictive value
        "npv",  # negative predictive value
        "jac",  # jaccard index, threat score, or critical success index
        "acc",  # accuracy
        "phi",  # phi coefficient, or matthews correlation coefficient
        "fnr",  # false negative rate, miss rate
        "fpr",  # false positive rate, fall-out
        "fdr",  # false discovery rate
        "for_",  # false omission rate
        "plr",  # positive likelihood ratio
        "nlr",  # negative likelihood ratio
        "pt",  # prevalence threshold
        "ba",  # balanced accuracy
        "f1",  # f1-measure
        "fm",  # fowlkes-mallows index
        "bm",  # bookmaker informedness
        "mk",  # markedness
        "dor",  # diagnostic odds ratio
        "auroc",  # area under the receiver-operater characteristic curve (ROC AUC)
        "auprc",  # area under the precision-recall curve
    ]:
        metrics[m.strip("_")] = eval(m)
        # convert to Python float from numpy float if possible
        metrics[f"macro_{m}".strip("_")] = (
            np.nanmean(eval(m) * _weights).item() if np.any(np.isfinite(eval(m))) else float("nan")
        )
    if fillna is not False:
        if isinstance(fillna, bool):
            fillna = 0.0
        assert 0 <= fillna <= 1, "fillna must be in [0, 1]"
        for m in metrics:
            if isinstance(metrics[m], np.ndarray):
                metrics[m][np.isnan(metrics[m])] = fillna
            elif np.isnan(metrics[m]):
                metrics[m] = fillna
    return metrics


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="F1-measure", metrics="F1-measures"),
    "prepend",
)
def f_measure(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
    thr: float = 0.5,
    fillna: Union[bool, float] = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_f1 : float
        Macro-averaged F1-measure.
    f1 : numpy.ndarray
        F1-measures for each class, of shape: ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights, thr, fillna)

    return m["macro_f1"], m["f1"]


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="sensitivity", metrics="sensitivities"),
    "prepend",
)
def sensitivity(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
    thr: float = 0.5,
    fillna: Union[bool, float] = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_sens : float
        Macro-averaged sensitivity.
    sens : numpy.ndarray
        Sensitivities for each class, of shape ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights, thr, fillna)

    return m["macro_sens"], m["sens"]


# aliases
recall = sensitivity
true_positive_rate = sensitivity
hit_rate = sensitivity


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="precision", metrics="precisions"),
    "prepend",
)
def precision(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
    thr: float = 0.5,
    fillna: Union[bool, float] = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_prec : float
        Macro-averaged precision.
    prec : numpy.ndarray
        Precisions for each class, of shape ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights, thr, fillna)

    return m["macro_prec"], m["prec"]


# aliases
positive_predictive_value = precision


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="specificity", metrics="specificities"),
    "prepend",
)
def specificity(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
    thr: float = 0.5,
    fillna: Union[bool, float] = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_spec : float
        Macro-averaged specificity.
    spec : numpy.ndarray
        Specificities for each class, of shape ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights, thr, fillna)

    return m["macro_spec"], m["spec"]


# aliases
selectivity = specificity
true_negative_rate = specificity


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="AUROC and macro AUPRC", metrics="AUPRCs, AUPRCs"),
    "prepend",
)
def auc(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
    thr: float = 0.5,
    fillna: Union[bool, float] = 0.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    macro_auroc : float
        Macro-averaged AUROC.
    macro_auprc : float
        Macro-averaged AUPRC.
    auprc : numpy.ndarray
        AUPRCs for each class, of shape ``(n_classes,)``.
    auprc : numpy.ndarray
        AUPRCs for each class, of shape ``(n_classes,)``.

    """
    if outputs.ndim == 1:
        raise ValueError("outputs must be of shape (n_samples, n_classes) to compute AUC")
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights, thr, fillna)

    return m["macro_auroc"], m["macro_auprc"], m["auroc"], m["auprc"]


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="accuracy", metrics="accuracies"),
    "prepend",
)
def accuracy(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
    thr: float = 0.5,
    fillna: Union[bool, float] = 0.0,
) -> float:
    """
    Returns
    -------
    macro_acc : float
        Macro-averaged accuracy.
    acc: numpy.ndarray,
        Accuracies for each class, of shape ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights, thr, fillna)

    return m["macro_acc"], m["acc"]


def QRS_score(
    rpeaks_truths: Sequence[Union[np.ndarray, Sequence[int]]],
    rpeaks_preds: Sequence[Union[np.ndarray, Sequence[int]]],
    fs: Real,
    thr: float = 0.075,
) -> float:
    """
    QRS accuracy score, proposed in CPSC2019.

    Parameters
    ----------
    rpeaks_truths : array_like
        array of ground truths of rpeaks locations (indices)
        from multiple records.
    rpeaks_preds : array_like
        predictions of ground truths of rpeaks locations (indices)
        for multiple records.
    fs : numbers.Real
        Sampling frequency of ECG signal
    thr : float, default 0.075
        Threshold for a prediction to be truth positive,
        with units in seconds.

    Returns
    -------
    rec_acc : float
        Accuracy of predictions.

    """
    assert len(rpeaks_truths) == len(
        rpeaks_preds
    ), f"number of records does not match, truth indicates {len(rpeaks_truths)}, while pred indicates {len(rpeaks_preds)}"
    n_records = len(rpeaks_truths)
    record_flags = np.ones((len(rpeaks_truths),), dtype=float)
    thr_ = thr * fs

    for idx, (truth_arr, pred_arr) in enumerate(zip(rpeaks_truths, rpeaks_preds)):
        false_negative = 0
        false_positive = 0
        true_positive = 0
        extended_truth_arr = np.concatenate((truth_arr.astype(int), [int(9.5 * fs)]))
        for j, t_ind in enumerate(extended_truth_arr[:-1]):
            next_t_ind = extended_truth_arr[j + 1]
            loc = np.where(np.abs(pred_arr - t_ind) <= thr_)[0]
            if j == 0:
                err = np.where((pred_arr >= 0.5 * fs + thr_) & (pred_arr <= t_ind - thr_))[0]
            else:
                err = np.array([], dtype=int)
            err = np.append(
                err,
                np.where((pred_arr >= t_ind + thr_) & (pred_arr <= next_t_ind - thr_))[0],
            )

            false_positive += len(err)
            if len(loc) >= 1:
                true_positive += 1
                false_positive += len(loc) - 1
            elif len(loc) == 0:
                false_negative += 1

        if false_negative + false_positive > 1:
            record_flags[idx] = 0
        elif false_negative == 1 and false_positive == 0:
            record_flags[idx] = 0.3
        elif false_negative == 0 and false_positive == 1:
            record_flags[idx] = 0.7

    rec_acc = round(np.sum(record_flags) / n_records, 4)

    return rec_acc


def one_hot_pair(
    labels: Union[np.ndarray, Tensor, Sequence[Sequence[int]]],
    outputs: Union[np.ndarray, Tensor, Sequence[Sequence[int]]],
    num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert categorical (of shape ``(n_samples,)``) labels and outputs
    to binary (of shape ``(n_samples, n_classes)``) labels and outputs if applicable.

    Parameters
    ----------
    labels : numpy.ndarray or torch.Tensor or Sequence[Sequence[int]]
        Categorical labels of shape ``(n_samples,)``,
        or categorical labels of length ``(n_samples,)``
        where each element is a sequence of class indices (multi-label),
        or binary labels of shape ``(n_samples, n_classes)``.
    outputs : numpy.ndarray or torch.Tensor or Sequence[Sequence[int]]
        Categorical outputs of shape ``(n_samples,)`` (single-label),
        or categorical outputs of length ``(n_samples,)``
        where each element is a sequence of class indices (multi-label),
        or binary outputs of shape ``(n_samples, n_classes)``.
    num_classes : int, optional
        Number of classes.
        Required if both `labels` and `outputs` are categorical.

    Returns
    -------
    labels : numpy.ndarray
        Binary labels of shape ``(n_samples, n_classes)``.
    outputs: numpy.ndarray
        Binary outputs of shape ``(n_samples, n_classes)``.

    """
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    if isinstance(outputs, Tensor):
        outputs = outputs.cpu().numpy()

    if num_classes is None:  # determine `num_classes`
        if isinstance(labels, np.ndarray) and labels.ndim == 2:
            num_classes = labels.shape[1]
        elif isinstance(outputs, np.ndarray) and outputs.ndim == 2:
            num_classes = outputs.shape[1]
    assert num_classes is not None, "num_classes is required if both labels and outputs are categorical"

    shape = (len(labels), num_classes)
    labels = _one_hot_pair(labels, shape)
    outputs = _one_hot_pair(outputs, shape)

    return labels, outputs


def _one_hot_pair(cls_array: Union[np.ndarray, Sequence[Sequence[int]]], shape: Tuple[int]) -> np.ndarray:
    """Convert categorical array to binary array.

    Parameters
    ----------
    cls_array : numpy.ndarray or Sequence[Sequence[int]]
        Categorical array of shape ``(n_samples,)``,
        where each element is a sequence of class indices (multi-label),
        or an integer of class index (single-label).
    shape: tuple,
        Shape of binary array.

    Returns
    -------
    numpy.ndarray
        Binarized array of `cls_array` of shape ``(n_samples, n_classes)``.

    """
    if isinstance(cls_array, np.ndarray) and cls_array.ndim == 2:
        return cls_array
    bin_array = np.zeros(shape)
    for i in range(shape[0]):
        bin_array[i, cls_array[i]] = 1
    return bin_array


@add_docstring(one_hot_pair.__doc__)
def cls_to_bin(
    labels: Union[np.ndarray, Tensor, Sequence[Sequence[int]]],
    outputs: Union[np.ndarray, Tensor, Sequence[Sequence[int]]],
    num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias of `one_hot_pair`."""
    warnings.warn("`cls_to_bin` is deprecated, use `one_hot_pair` instead", DeprecationWarning)
    return one_hot_pair(labels, outputs, num_classes)


def compute_wave_delineation_metrics(
    truth_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    class_map: Dict[str, int],
    fs: Real,
    mask_format: str = "channel_first",
    tol: Real = 0.15,
) -> Dict[str, Dict[str, float]]:
    f"""
    Compute metrics for the task of ECG wave delineation
    (sensitivity, precision, f1_score, mean error and standard deviation of the mean errors)
    for multiple evaluations.

    Parameters
    ----------
    truth_masks : Sequence[numpy.ndarray]
        A sequence of ground truth masks,
        each of which can also hold multiple masks from different samples
        (differ by record or by lead).
        Each mask is of shape ``(n_channels, n_timesteps)``
        or ``(n_timesteps, n_channels)``.
    pred_masks : Sequence[numpy.ndarray]
        Predictions corresponding to `truth_masks`,
        of the same shapes.
    class_map : dict
        Class map, mapping names to waves to numbers from 0 to n_classes-1,
        the keys should contain {", ".join([f'"{item}"' for item in ECGWaveFormNames])}.
    fs : numbers.Real
        Sampling frequency of the signal corresponding to the masks,
        used to compute the duration of each waveform,
        and thus the error and standard deviations of errors.
    mask_format : str, default "channel_first"
        Format of the mask, one of the following:
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first")
    tol : float, default 0.15
        Tolerance for the duration of the waveform,
        with units in seconds.

    Returns
    -------
    scorings : dict
        scorings of onsets and offsets of pwaves, qrs complexes, twaves.
        Each scoring is a dict consisting of the following metrics:
        sensitivity, precision, f1_score, mean_error, standard_deviation

    """
    assert len(truth_masks) == len(pred_masks), "length of truth_masks and pred_masks should be the same"
    truth_waveforms, pred_waveforms = [], []
    # compute for each element
    for tm, pm in zip(truth_masks, pred_masks):
        n_masks = tm.shape[0] if mask_format.lower() in ["channel_first", "lead_first"] else tm.shape[1]

        new_t = masks_to_waveforms(tm, class_map, fs, mask_format)
        new_t = [new_t[f"lead_{idx+1}"] for idx in range(n_masks)]  # list of list of `ECGWaveForm`s
        truth_waveforms += new_t

        new_p = masks_to_waveforms(pm, class_map, fs, mask_format)
        new_p = [new_p[f"lead_{idx+1}"] for idx in range(n_masks)]  # list of list of `ECGWaveForm`s
        pred_waveforms += new_p

    scorings = compute_metrics_waveform(truth_waveforms, pred_waveforms, fs, tol)

    return scorings


def compute_metrics_waveform(
    truth_waveforms: Sequence[Sequence[ECGWaveForm]],
    pred_waveforms: Sequence[Sequence[ECGWaveForm]],
    fs: Real,
    tol: Real = 0.15,
) -> Dict[str, Dict[str, float]]:
    """
    compute the sensitivity, precision, f1_score, mean error
    and standard deviation of the mean errors,
    of evaluations on a multiple samples (differ by records, or leads).

    Parameters
    ----------
    truth_waveforms : Sequence[Sequence[ECGWaveForm]]
        The ground truth,
        each element is a sequence of `ECGWaveForm` from the same sample.
    pred_waveforms : Sequence[Sequence[ECGWaveForm]]
        The predictions corresponding to `truth_waveforms`,
        each element is a sequence of :class:`ECGWaveForm` from the same sample.
    fs : numbers.Real
        Sampling frequency of the signal corresponding to the waveforms,
        used to compute the duration of each waveform,
        and thus the error and standard deviations of errors.
    tol : float, default 0.15
        Tolerance for the duration of the waveform,
        with units in seconds.

    Returns
    -------
    scorings : dict
        Scorings of onsets and offsets of pwaves, qrs complexes, twaves.
        Each scoring is a dict consisting of the following metrics:
        sensitivity, precision, f1_score, mean_error, standard_deviation.

    """
    truth_positive = dict({f"{wave}_{term}": 0 for wave in ECGWaveFormNames for term in ["onset", "offset"]})
    false_positive = dict({f"{wave}_{term}": 0 for wave in ECGWaveFormNames for term in ["onset", "offset"]})
    false_negative = dict({f"{wave}_{term}": 0 for wave in ECGWaveFormNames for term in ["onset", "offset"]})
    errors = dict({f"{wave}_{term}": [] for wave in ECGWaveFormNames for term in ["onset", "offset"]})
    # accumulating results
    for tw, pw in zip(truth_waveforms, pred_waveforms):
        s = _compute_metrics_waveform(tw, pw, fs, tol)
        for wave in [
            "pwave",
            "qrs",
            "twave",
        ]:
            for term in ["onset", "offset"]:
                truth_positive[f"{wave}_{term}"] += s[f"{wave}_{term}"]["truth_positive"]
                false_positive[f"{wave}_{term}"] += s[f"{wave}_{term}"]["false_positive"]
                false_negative[f"{wave}_{term}"] += s[f"{wave}_{term}"]["false_negative"]
                errors[f"{wave}_{term}"] += s[f"{wave}_{term}"]["errors"]
    scorings = dict()
    for wave in ECGWaveFormNames:
        for term in ["onset", "offset"]:
            tp = truth_positive[f"{wave}_{term}"]
            fp = false_positive[f"{wave}_{term}"]
            fn = false_negative[f"{wave}_{term}"]
            err = errors[f"{wave}_{term}"]
            sensitivity = tp / (tp + fn + DEFAULTS.eps)
            precision = tp / (tp + fp + DEFAULTS.eps)
            f1_score = 2 * sensitivity * precision / (sensitivity + precision + DEFAULTS.eps)
            mean_error = np.mean(err) * 1000 / fs if len(err) > 0 else np.nan
            standard_deviation = np.std(err) * 1000 / fs if len(err) > 0 else np.nan
            scorings[f"{wave}_{term}"] = dict(
                sensitivity=sensitivity,
                precision=precision,
                f1_score=f1_score,
                mean_error=mean_error,
                standard_deviation=standard_deviation,
            )

    return scorings


def _compute_metrics_waveform(
    truths: Sequence[ECGWaveForm],
    preds: Sequence[ECGWaveForm],
    fs: Real,
    tol: Real = 0.15,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the sensitivity, precision, f1_score, mean error
    and standard deviation of the mean errors,
    of evaluations on a single sample (the same record, the same lead)

    Parameters
    ----------
    truths : Sequence[ECGWaveForm]
        The ground truth
    preds : Sequence[ECGWaveForm]
        The predictions corresponding to `truths`.
    fs : numbers.Real
        Sampling frequency of the signal corresponding to the waveforms,
        used to compute the duration of each waveform,
        and thus the error and standard deviations of errors.
    tol : float, default 0.15,
        Tolerance for the duration of the waveform,
        with units in seconds.

    Returns
    -------
    scorings : dict
        Scorings of onsets and offsets of pwaves, qrs complexes, twaves.
        Each scoring is a dict consisting of the following metrics:
        truth_positive, false_negative, false_positive, errors,
        sensitivity, precision, f1_score, mean_error, standard_deviation

    """
    pwave_onset_truths, pwave_offset_truths, pwave_onset_preds, pwave_offset_preds = (
        [],
        [],
        [],
        [],
    )
    qrs_onset_truths, qrs_offset_truths, qrs_onset_preds, qrs_offset_preds = (
        [],
        [],
        [],
        [],
    )
    twave_onset_truths, twave_offset_truths, twave_onset_preds, twave_offset_preds = (
        [],
        [],
        [],
        [],
    )

    for item in ["truths", "preds"]:
        for w in eval(item):
            for term in ["onset", "offset"]:
                eval(f"{w.name}_{term}_{item}.append(w.{term})")

    scorings = dict()
    for wave in ECGWaveFormNames:
        for term in ["onset", "offset"]:
            (
                truth_positive,
                false_negative,
                false_positive,
                errors,
                sensitivity,
                precision,
                f1_score,
                mean_error,
                standard_deviation,
            ) = _compute_metrics_base(eval(f"{wave}_{term}_truths"), eval(f"{wave}_{term}_preds"), fs, tol)
            scorings[f"{wave}_{term}"] = dict(
                truth_positive=truth_positive,
                false_negative=false_negative,
                false_positive=false_positive,
                errors=errors,
                sensitivity=sensitivity,
                precision=precision,
                f1_score=f1_score,
                mean_error=mean_error,
                standard_deviation=standard_deviation,
            )
    return scorings


def _compute_metrics_base(truths: Sequence[Real], preds: Sequence[Real], fs: Real, tol: Real = 0.15) -> Dict[str, float]:
    r"""Base function for computing the metrics of the onset and offset of a waveform.

    Parameters
    ----------
    truths : Sequence[numbers.Real]
        Ground truth of indices of corresponding critical points.
    preds : Sequence[numbers.Real]
        Predicted indices of corresponding critical points.
    fs : numbers.Real
        Sampling frequency of the signal corresponding to the critical points,
        used to compute the duration of each waveform,
        and thus the error and standard deviations of errors.
    tol : float, default 0.15
        Tolerance for the duration of the waveform,
        with units in seconds.

    Returns
    -------
    tuple
        Tuple of metrics:
        truth_positive, false_negative, false_positive, errors,
        sensitivity, precision, f1_score, mean_error, standard_deviation.
        See [#dl_seg]_ for more details.

    References
    ----------
    .. [#dl_seg] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov.
                 "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics.
                 Springer, Cham, 2019.

    """
    _tolerance = round(tol * fs)
    _truths = np.array(truths)
    _preds = np.array(preds)
    truth_positive, false_positive, false_negative = 0, 0, 0
    errors = []
    n_included = 0
    for point in truths:
        _pred = _preds[np.where(np.abs(_preds - point) <= _tolerance)[0].tolist()]
        if len(_pred) > 0:
            truth_positive += 1
            idx = np.argmin(np.abs(_pred - point))
            errors.append(_pred[idx] - point)
        else:
            false_negative += 1
        n_included += len(_pred)

    false_positive = len(_preds) - truth_positive

    sensitivity = truth_positive / (truth_positive + false_negative + DEFAULTS.eps)
    precision = truth_positive / (truth_positive + false_positive + DEFAULTS.eps)
    f1_score = 2 * sensitivity * precision / (sensitivity + precision + DEFAULTS.eps)
    mean_error = np.mean(errors) * 1000 / fs if len(errors) > 0 else np.nan
    standard_deviation = np.std(errors) * 1000 / fs if len(errors) > 0 else np.nan

    return (
        truth_positive,
        false_negative,
        false_positive,
        errors,
        sensitivity,
        precision,
        f1_score,
        mean_error,
        standard_deviation,
    )
