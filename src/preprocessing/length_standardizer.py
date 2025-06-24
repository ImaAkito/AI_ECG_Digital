import numpy as np
from typing import Tuple

class ECGLengthStandardizer: #Class for standardizing the length of the ECG signal (cropping or padding with zeros)
    def __init__(self, target_length: int):
        self.target_length = target_length

    def standardize(self, signal: np.ndarray) -> np.ndarray: #function for standardizing the length of the ECG signal (cropping or padding with zeros)
        if signal.ndim == 1:
            return self._fix_length_1d(signal)
        elif signal.ndim == 2:
            return np.vstack([self._fix_length_1d(ch) for ch in signal])
        else:
            raise ValueError('signal must be 1D or 2D array')

    def _fix_length_1d(self, arr: np.ndarray) -> np.ndarray:
        n = len(arr)
        if n == self.target_length:
            return arr.copy()
        elif n > self.target_length:
            start = (n - self.target_length) // 2
            return arr[start:start + self.target_length]
        else:
            pad_left = (self.target_length - n) // 2
            pad_right = self.target_length - n - pad_left
            return np.pad(arr, (pad_left, pad_right), mode='constant') 