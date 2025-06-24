import numpy as np
from scipy.signal import resample
from typing import Optional

class ECGResampler: #Class for changing the sampling frequency of the ECG signal
    def __init__(self, original_fs: int, target_fs: int):
        self.original_fs = original_fs
        self.target_fs = target_fs

    def resample_signal(self, signal: np.ndarray) -> np.ndarray: #function for resampling the ECG signal to a new sampling frequency
        if self.original_fs == self.target_fs:
            return signal.copy()
        factor = self.target_fs / self.original_fs
        new_len = int(round(signal.shape[-1] * factor))
        if signal.ndim == 1:
            return resample(signal, new_len)
        elif signal.ndim == 2:
            return np.vstack([resample(ch, new_len) for ch in signal])
        else:
            raise ValueError('signal must be 1D or 2D array') 