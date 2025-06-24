from .noise_augmentor import ECGNoiseAugmentor
from .channel_dropout import ECGChannelDropout
from .scaling_augmentor import ECGScalingAugmentor
from .time_shift_augmentor import ECGTimeShiftAugmentor
from .rare_class_generator import RareECGClassGenerator
import numpy as np
from typing import Optional, Dict

class ECGAugmentationPipeline: #Main class for ECG augmentation pipeline. Allows combining noise, dropout, scaling, shifting and rare class generation.
    def __init__(self,
                 noise_params: Optional[dict] = None,
                 dropout_params: Optional[dict] = None,
                 scaling_params: Optional[dict] = None,
                 shift_params: Optional[dict] = None):
        self.noise_aug = ECGNoiseAugmentor(**(noise_params or {}))
        self.dropout_aug = ECGChannelDropout(**(dropout_params or {}))
        self.scaling_aug = ECGScalingAugmentor(**(scaling_params or {}))
        self.shift_aug = ECGTimeShiftAugmentor(**(shift_params or {}))
        self.rare_generator = None

    def set_rare_class_templates(self, templates: Dict[str, np.ndarray]):
        self.rare_generator = RareECGClassGenerator(templates)

    def augment(self, signal: np.ndarray, random_state: Optional[int] = None) -> np.ndarray:
        x = signal.copy()
        x = self.noise_aug.augment(x, random_state)
        x = self.dropout_aug.augment(x, random_state)
        x = self.scaling_aug.augment(x, random_state)
        x = self.shift_aug.augment(x, random_state)
        return x

    def generate_rare(self, class_name: str, n: int = 10, random_state: Optional[int] = None) -> np.ndarray:
        if self.rare_generator is None:
            raise RuntimeError('RareECGClassGenerator is not initialized. Call set_rare_class_templates.')
        return self.rare_generator.generate(class_name, n, random_state) 