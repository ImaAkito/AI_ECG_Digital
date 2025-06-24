import numpy as np
from typing import Optional

class ECGScalingAugmentor: #Augmentation method for ECG by scaling of amplitudes of channels
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def augment(self, signal: np.ndarray, random_state: Optional[int] = None) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)
        scales = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(signal.shape[0], 1))
        return signal * scales 