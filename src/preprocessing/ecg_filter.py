import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import Optional, Tuple

class ECGSignalFilter: #bandpass filtering of the ECG signal using a Butterworth filter
    def __init__(self, sample_rate: int, low_freq: float = 0.5, high_freq: float = 40.0, order: int = 4):
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self._sos = self._design_filter()

    def _design_filter(self) -> np.ndarray:
        nyquist = 0.5 * self.sample_rate
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        if low <= 0 and high >= 1:
            raise ValueError("Incorrect filter frequencies: check low_freq and high_freq.")
        sos = butter(self.order, [low, high], btype='band', output='sos')
        return sos

    def apply(self, signal: np.ndarray) -> np.ndarray: #Applying a filter to a signal (support for 1D and 2D: (n_leads, siglen))
        if signal.ndim == 1:
            return sosfiltfilt(self._sos, signal)
        elif signal.ndim == 2:
            return np.vstack([sosfiltfilt(self._sos, ch) for ch in signal])
        else:
            raise ValueError("signal must be 1D or 2D array")


def bandpass_filter_ecg( #functuon for fast bandpass filtering of the ECG signal
    ecg: np.ndarray,
    fs: int,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4
) -> np.ndarray:
    filter_instance = ECGSignalFilter(sample_rate=fs, low_freq=low, high_freq=high, order=order)
    return filter_instance.apply(ecg) 