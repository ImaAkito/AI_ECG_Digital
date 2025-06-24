import numpy as np
from typing import Optional

class ECGChannelDropout: #Augmentation method for ECG by random channel dropout 
    def __init__(self, dropout_prob: float = 0.2):
        self.dropout_prob = dropout_prob

    def augment(self, signal: np.ndarray, random_state: Optional[int] = None) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)
        mask = np.random.rand(signal.shape[0]) > self.dropout_prob
        out = signal.copy()
        for i, keep in enumerate(mask):
            if not keep:
                out[i, :] = 0.0
        return out 