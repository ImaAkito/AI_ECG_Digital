import numpy as np
from typing import Literal, Optional

class ECGNoiseAugmentor: #Augmentation method for ECG by noise of different types   
    def __init__(self, noise_type: Literal['gaussian', 'pink', 'electrode'] = 'gaussian', std: float = 0.01):
        self.noise_type = noise_type
        self.std = std

    def augment(self, signal: np.ndarray, random_state: Optional[int] = None) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.std, size=signal.shape)
        elif self.noise_type == 'pink':
            noise = self._pink_noise(signal.shape)
            noise = noise * self.std / (np.std(noise) + 1e-8)
        elif self.noise_type == 'electrode':
            noise = self._electrode_noise(signal.shape)
            noise = noise * self.std / (np.std(noise) + 1e-8)
        else:
            raise ValueError('Unknown noise type')
        return signal + noise

    def _pink_noise(self, shape): #Simple implementation of pink noise (1/f)
        n = np.prod(shape)
        uneven = n % 2
        X = np.random.randn(n // 2 + 1 + uneven) + 1j * np.random.randn(n // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)
        y = (np.fft.irfft(X / S)).real
        y = y[:n].reshape(shape)
        return y

    def _electrode_noise(self, shape):  #Simulate electrode noise as a sine wave with low frequency + drift
        t = np.linspace(0, 1, shape[-1])
        noise = 0.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz
        drift = 0.1 * np.sin(2 * np.pi * 0.5 * t)
        noise = noise + drift
        if len(shape) == 2:
            noise = np.tile(noise, (shape[0], 1))
        return noise 