import numpy as np
from typing import Dict, Any

class ECGQualityAssessor: #Class for assessing the quality of the ECG signal (flatline, outliers, NaN, noise)       
    def __init__(self, flatline_thr: float = 1e-3, nan_thr: float = 0.01, outlier_thr: float = 5.0):
        self.flatline_thr = flatline_thr
        self.nan_thr = nan_thr
        self.outlier_thr = outlier_thr

    def assess(self, signal: np.ndarray) -> Dict[str, Any]: #function for assessing the quality of the ECG signal (flatline, outliers, NaN, noise)
        result = {}
        if signal.ndim == 1:
            result = self._assess_1d(signal)
        elif signal.ndim == 2:
            for i in range(signal.shape[0]):
                result[f'lead_{i+1}'] = self._assess_1d(signal[i])
        else:
            raise ValueError('signal must be 1D or 2D array')
        return result

    def _assess_1d(self, arr: np.ndarray) -> Dict[str, Any]: #function for assessing the quality of the ECG signal (flatline, outliers, NaN, noise)     
        n = len(arr)
        n_nan = np.sum(np.isnan(arr))
        nan_ratio = n_nan / n
        is_nan = nan_ratio > self.nan_thr
        is_flat = np.max(arr) - np.min(arr) < self.flatline_thr
        n_outliers = np.sum(np.abs(arr - np.median(arr)) > self.outlier_thr * np.std(arr))
        return {
            'nan_ratio': nan_ratio,
            'is_nan': is_nan,
            'is_flat': is_flat,
            'n_outliers': int(n_outliers),
        } 