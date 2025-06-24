import numpy as np
from typing import Optional

class ECGGapFiller: #Class for restoring gaps (NaN) and processing flat sections in ECG
    def __init__(self, flat_thr: float = 1e-3):
        self.flat_thr = flat_thr

    def fill_gaps(self, signal: np.ndarray) -> np.ndarray: #function for linear interpolation of NaN values
        if signal.ndim == 1:
            return self._interp_1d(signal)
        elif signal.ndim == 2:
            return np.vstack([self._interp_1d(ch) for ch in signal])
        else:
            raise ValueError('signal must be 1D or 2D array')

    def _interp_1d(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        nans = np.isnan(arr)
        if np.all(nans): #if all values are NaN, fill with zeros (can be changed to another value if needed)
            arr[:] = 0.0
        elif np.any(nans):
            arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
        return arr

    def fix_flatlines(self, signal: np.ndarray) -> np.ndarray: #function for replacing flat sections with NaN to then interpolate
        if signal.ndim == 1:
            return self._flat_to_nan(signal)
        elif signal.ndim == 2:
            return np.vstack([self._flat_to_nan(ch) for ch in signal])
        else:
            raise ValueError('signal must be 1D or 2D array')

    def _flat_to_nan(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        if np.max(arr) - np.min(arr) < self.flat_thr:
            arr[:] = np.nan
        return arr 