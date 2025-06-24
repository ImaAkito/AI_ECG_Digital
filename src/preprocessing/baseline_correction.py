import numpy as np
from scipy.signal import medfilt
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Optional

class BaselineDriftRemover: #Removing baseline drift from ECGs using median filters
    def __init__(self, fs: int, short_win: float = 0.2, long_win: float = 0.6):
        self.fs = fs
        self.short_win = short_win
        self.long_win = long_win

    def remove(self, signal: np.ndarray) -> np.ndarray: #Removing baseline drift using two median filters (short and long window).
        win1 = int(self.short_win * self.fs)
        win2 = int(self.long_win * self.fs)
        if win1 % 2 == 0:
            win1 += 1
        if win2 % 2 == 0:
            win2 += 1
        if signal.ndim == 1:
            baseline = medfilt(signal, win1)
            baseline = medfilt(baseline, win2)
            return signal - baseline
        elif signal.ndim == 2:
            result = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                baseline = medfilt(signal[i], win1)
                baseline = medfilt(baseline, win2)
                result[i] = signal[i] - baseline
            return result
        else:
            raise ValueError("signal must be 1D or 2D array")


def loess_baseline_correction(signal: np.ndarray, frac: float = 0.02) -> np.ndarray: #Removing baseline drift using LOESS regression (smooth approximation)
    if signal.ndim == 1:
        baseline = lowess(signal, np.arange(len(signal)), frac=frac, return_sorted=False)
        return signal - baseline
    elif signal.ndim == 2:
        result = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            baseline = lowess(signal[i], np.arange(signal.shape[1]), frac=frac, return_sorted=False)
            result[i] = signal[i] - baseline
        return result
    else:
        raise ValueError("signal must be 1D or 2D array") 