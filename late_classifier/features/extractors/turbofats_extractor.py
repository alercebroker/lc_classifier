from typing import List

from late_classifier.features.core.base import FeatureExtractorSingleBand
from turbofats import NewFeatureSpace
import pandas as pd
import logging


class TurboFatsFeatureExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        self.feature_space = NewFeatureSpace(self.get_features_keys())

    def get_features_keys(self) -> List[str]:
        return [
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

    def get_required_keys(self) -> List[str]:
        return ['mjd', 'magpsf_corr', 'fid', 'sigmapsf_corr']

    def compute_feature_in_one_band(self, detections, band=None, **kwargs):
        """
        Compute features for detections

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        band :class:int
        kwargs

        Returns class:pandas.`DataFrame`
        Turbo FATS features.
        -------

        """
        index = detections.index.unique()[0]
        columns = self.get_features_keys_with_band(band)
        mag = [f"Harmonics_mag_{i}_{band}" for i in range(1, 8)]
        phase = [f"Harmonics_phase_{i}_{band}" for i in range(2, 8)]
        columns = columns + mag + phase + [f"Harmonics_mse_{band}"]
        detections = detections[detections.fid == band]

        if band is None or len(detections) == 0:
            logging.warning(f'extractor=TURBOFATS  object={index}  filters_qty=1')
            return pd.DataFrame(columns=columns, index=[index])

        df = self.feature_space.calculate_features(detections)
        df.columns = [f'{x}_{band}' for x in df.columns]
        return df
