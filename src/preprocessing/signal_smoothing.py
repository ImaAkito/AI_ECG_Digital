import numpy as np
from scipy.signal import medfilt, get_window
from typing import Literal

class ECGSmoother: #Class for smoothing the ECG signal by different methods (moving average, median, Blackman and др.)
    def __init__(self, method: Literal['mean', 'median', 'blackman'] = 'mean', window_size: int = 11):
        self.method = method
        self.window_size = window_size

    def smooth(self, signal: np.ndarray) -> np.ndarray: #function for smoothing the ECG signal by different methods (moving average, median, Blackman and др.)
        if self.method == 'mean':
            return self._moving_average(signal)
        elif self.method == 'median':
            return self._median_filter(signal)
        elif self.method == 'blackman':
            return self._window_filter(signal, 'blackman')
        else:
            raise ValueError(f'Unknown smoothing method: {self.method}')

    def _moving_average(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return np.convolve(arr, np.ones(self.window_size)/self.window_size, mode='same')
        elif arr.ndim == 2:
            return np.vstack([np.convolve(ch, np.ones(self.window_size)/self.window_size, mode='same') for ch in arr])
        else:
            raise ValueError('signal must be 1D or 2D array')

    def _median_filter(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return medfilt(arr, self.window_size)
        elif arr.ndim == 2:
            return np.vstack([medfilt(ch, self.window_size) for ch in arr])
        else:
            raise ValueError('signal must be 1D or 2D array')

    def _window_filter(self, arr: np.ndarray, window_type: str) -> np.ndarray:
        win = get_window(window_type, self.window_size)
        win = win / np.sum(win)
        if arr.ndim == 1:
            return np.convolve(arr, win, mode='same')
        elif arr.ndim == 2:
            return np.vstack([np.convolve(ch, win, mode='same') for ch in arr])
        else:
            raise ValueError('signal must be 1D or 2D array') 