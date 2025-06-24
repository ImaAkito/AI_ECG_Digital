import numpy as np
from typing import Dict, Any, Literal
from .ecg_filter import ECGSignalFilter
from .baseline_correction import BaselineDriftRemover
from .signal_normalizer import ECGNormalizer
from .length_standardizer import ECGLengthStandardizer
from .resampler import ECGResampler
from .polarity_check import ECGPolarityChecker
from .signal_quality import ECGQualityAssessor
from .gap_filler import ECGGapFiller
from .signal_smoothing import ECGSmoother

class ECGPreprocessingPipeline: #Universal preprocessing pipeline for ECG signal with flexible parameters for all stages
    def __init__(
        self,
        orig_fs: int = 500,
        target_fs: int = 500,
        target_len: int = 5000,
        filter_low: float = 0.5,
        filter_high: float = 40.0,
        filter_order: int = 4,
        baseline_short: float = 0.2,
        baseline_long: float = 0.6,
        smooth_method: Literal['mean', 'median', 'blackman'] = 'mean',
        smooth_window: int = 11,
        polarity_window_sec: float = 2.0,
        normalizer_method: Literal['zscore', 'minmax', 'custom'] = 'zscore',
        gap_flat_thr: float = 1e-3,
        quality_flat_thr: float = 1e-3,
        quality_nan_thr: float = 0.01,
        quality_outlier_thr: float = 5.0,
    ):
        self.resampler = ECGResampler(orig_fs, target_fs)
        self.filter = ECGSignalFilter(sample_rate=target_fs, low_freq=filter_low, high_freq=filter_high, order=filter_order)
        self.baseline = BaselineDriftRemover(fs=target_fs, short_win=baseline_short, long_win=baseline_long)
        self.smoother = ECGSmoother(method=smooth_method, window_size=smooth_window)
        self.polarity = ECGPolarityChecker(window_sec=polarity_window_sec, fs=target_fs)
        self.gap_filler = ECGGapFiller(flat_thr=gap_flat_thr)
        self.normalizer = ECGNormalizer(method=normalizer_method)
        self.length_std = ECGLengthStandardizer(target_length=target_len)
        self.quality = ECGQualityAssessor(flatline_thr=quality_flat_thr, nan_thr=quality_nan_thr, outlier_thr=quality_outlier_thr)

    def process(self, signal: np.ndarray) -> Dict[str, Any]: #function for applying all preprocessing stages to the signal and returning the result and metadata
        # 1. Resampling
        sig = self.resampler.resample_signal(signal)
        # 2. Filtering
        sig = self.filter.apply(sig)
        # 3. Removing baseline
        sig = self.baseline.remove(sig)
        # 4. Smoothing
        sig = self.smoother.smooth(sig)
        # 5. Polarity correction
        sig = self.polarity.correct(sig)
        # 6. Processing flat sections and NaN
        sig = self.gap_filler.fix_flatlines(sig)
        sig = self.gap_filler.fill_gaps(sig)
        # 7. Normalization
        sig = self.normalizer.normalize(sig)
        # 8. Standardizing to the target length
        sig = self.length_std.standardize(sig)
        # 9. Quality assessment
        quality = self.quality.assess(sig)
        return {'signal': sig, 'quality': quality} 