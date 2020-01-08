from late_classifier.features.core.base import FeatureExtractorSingleBand
from turbofats import NewFeatureSpace


class TurboFatsFeatureExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        self.features_keys = [
            'Amplitude', 'AndersonDarling', 'Autocor_length',
            'Beyond1Std',
            'Con', 'Eta_e',
            'Gskew',
            'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
            'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31',
            'PeriodLS_v2',
            'Period_fit_v2', 'Psi_CS_v2', 'Psi_eta_v2', 'Rcs',
            'Skew', 'SmallKurtosis', 'Std',
            'StetsonK', 'Harmonics',
            'Pvar', 'ExcessVar',
            'GP_DRW_sigma', 'GP_DRW_tau', 'SF_ML_amplitude', 'SF_ML_gamma',
            'IAR_phi',
            'LinearTrend',
            'PeriodPowerRate'
        ]
        self.feature_space = NewFeatureSpace(self.features_keys)

    def _compute_features(self, detections, **kwargs):
        """
        Compute features for detections

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        kwargs

        Returns class:pandas.`DataFrame`
        Turbo FATS features.
        -------

        """
        return self.feature_space.calculate_features(detections)
