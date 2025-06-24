import numpy as np
from typing import Tuple

class ECGPolarityChecker: #Class for checking and correcting the inversion of the ECG signal
    def __init__(self, window_sec: float = 2.0, fs: int = 500):
        self.window_sec = window_sec
        self.fs = fs

    def is_inverted(self, signal: np.ndarray) -> bool: #function for checking if the signal is inverted (by median of the maximum square of the window)
        win = int(self.window_sec * self.fs)
        if signal.ndim == 1:
            return self._check_inversion(signal, win)
        elif signal.ndim == 2: #checking the inversion of the signal by the median of the maximum square of the window
            return self._check_inversion(signal[0], win)
        else:
            raise ValueError('signal must be 1D or 2D array')

    def _check_inversion(self, arr: np.ndarray, win: int) -> bool:
        arr = arr - np.mean(arr)
        max_sq = [np.max(arr[i:i+win]**2) for i in range(0, len(arr)-win, win)]
        med = np.median(max_sq)
        return med < 0

    def correct(self, signal: np.ndarray) -> np.ndarray: #function for correcting the inversion of the signal
        if self.is_inverted(signal):
            return -signal
        return signal 