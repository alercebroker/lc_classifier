from late_classifier.features.core.base import FeatureExtractorSingleBand
from turbofats import NewFeatureSpace
import pandas as pd
import logging


class TurboFatsFeatureExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        super().__init__()
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

    def _compute_features(self, detections, band=None, **kwargs):
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
        index = detections.index.unique()[0]
        columns = self.get_features_keys(band)
        mag = [f"Harmonics_mag_{i}_{band}" for i in range(1, 8)]
        phase = [f"Harmonics_phase_{i}_{band}" for i in range(2, 8)]
        columns = columns + mag + phase + [f"Harmonics_mse_{band}"]
        detections = detections[detections.fid == band]

        if band is None or len(detections) == 0:
            logging.error(
                f'TURBOFATS: Input dataframe invalid {index}\n - Required one filter.')
            return pd.DataFrame(columns=columns, index=[index])

        df =  self.feature_space.calculate_features(detections)
        df.columns = [f'{x}_{band}' for x in df.columns]
        return df
