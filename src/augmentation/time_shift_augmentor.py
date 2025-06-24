import numpy as np
from typing import Optional

class ECGTimeShiftAugmentor: #Augmentation method for ECG by random cyclic time shift
    def __init__(self, max_shift: int = 100):
        self.max_shift = max_shift

    def augment(self, signal: np.ndarray, random_state: Optional[int] = None) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)
        shifts = np.random.randint(-self.max_shift, self.max_shift + 1, size=signal.shape[0])
        out = np.zeros_like(signal)
        for i, shift in enumerate(shifts):
            out[i] = np.roll(signal[i], shift)
        return out 