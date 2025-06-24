import numpy as np
from typing import Literal, Union

class ECGNormalizer: #Class for normalizing the ECG signal by different methods (z-score, min-max, custom)  
    def __init__(self, method: Literal['zscore', 'minmax', 'custom'] = 'zscore'):
        self.method = method

    def normalize(self, signal: np.ndarray, mean: Union[float, np.ndarray] = 0.0, std: Union[float, np.ndarray] = 1.0) -> np.ndarray: #function for normalizing the ECG signal by different methods (z-score, min-max, custom)
        if self.method == 'zscore':
            if signal.ndim == 1:
                m = np.mean(signal)
                s = np.std(signal)
                return (signal - m) / (s + 1e-8)
            elif signal.ndim == 2:
                m = np.mean(signal, axis=1, keepdims=True)
                s = np.std(signal, axis=1, keepdims=True)
                return (signal - m) / (s + 1e-8)
            else:
                raise ValueError('signal must be 1D or 2D array')
        elif self.method == 'minmax':
            if signal.ndim == 1:
                minv = np.min(signal)
                maxv = np.max(signal)
                return (signal - minv) / (maxv - minv + 1e-8)
            elif signal.ndim == 2:
                minv = np.min(signal, axis=1, keepdims=True)
                maxv = np.max(signal, axis=1, keepdims=True)
                return (signal - minv) / (maxv - minv + 1e-8)
            else:
                raise ValueError('signal must be 1D or 2D array')
        elif self.method == 'custom':
            return (signal - mean) / (std + 1e-8)
        else:
            raise ValueError(f'Unknown normalization method: {self.method}') 