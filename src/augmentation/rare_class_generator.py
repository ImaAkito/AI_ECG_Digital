import numpy as np
from typing import Optional, Dict
from .noise_augmentor import ECGNoiseAugmentor
from .scaling_augmentor import ECGScalingAugmentor
from .time_shift_augmentor import ECGTimeShiftAugmentor

class RareECGClassGenerator: #Class for generation of synthetic ECGs of rare classes based on templates and augmentations
    def __init__(self, templates: Dict[str, np.ndarray]): #templates: dict, where key is class name and value is averaged signal (n_leads, siglen)
        self.templates = templates
        self.noise_aug = ECGNoiseAugmentor(noise_type='gaussian', std=0.02)
        self.scale_aug = ECGScalingAugmentor(scale_range=(0.85, 1.15))
        self.shift_aug = ECGTimeShiftAugmentor(max_shift=80)

    def generate(self, class_name: str, n: int = 10, random_state: Optional[int] = None) -> np.ndarray:
        if class_name not in self.templates:
            raise ValueError(f'No template for class {class_name}')
        base = self.templates[class_name]
        signals = []
        for i in range(n):
            sig = base.copy()
            sig = self.noise_aug.augment(sig, random_state)
            sig = self.scale_aug.augment(sig, random_state)
            sig = self.shift_aug.augment(sig, random_state)
            # warping (simple implementation: random stretching/compression in time)
            if np.random.rand() < 0.5:
                sig = self._random_warp(sig)
            signals.append(sig)
        return np.stack(signals)

    def _random_warp(self, signal: np.ndarray) -> np.ndarray:
        # Simple nonlinear time distortion
        from scipy.interpolate import interp1d
        n_leads, siglen = signal.shape
        t = np.linspace(0, 1, siglen)
        warp = t + 0.03 * np.sin(2 * np.pi * t * np.random.uniform(1, 5))
        warp = np.clip(warp, 0, 1)
        out = np.zeros_like(signal)
        for i in range(n_leads):
            f = interp1d(t, signal[i], kind='cubic', fill_value='extrapolate')
            out[i] = f(warp)
        return out 