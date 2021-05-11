from typing import Tuple, List
from functools import lru_cache

from ..core.base import FeatureExtractorSingleBand
from turbofats import FeatureSpace
import pandas as pd
import logging


class TurboFatsFeatureExtractor(FeatureExtractorSingleBand):
    def __init__(self, bands: List):
        super().__init__(bands)
        self.feature_space = FeatureSpace(
            self._feature_keys_for_new_feature_space())
        self.feature_space.data_column_names = [
            'magnitude', 'time', 'error']

    @lru_cache(1)
    def _feature_keys_for_new_feature_space(self):
        return (
            'Amplitude', 'AndersonDarling', 'Autocor_length',
            'Beyond1Std',
            'Con', 'Eta_e',
            'Gskew',
            'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
            'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31',
            'Rcs',
            'Skew', 'SmallKurtosis', 'Std',
            'StetsonK',
            'Pvar', 'ExcessVar',
            'SF_ML_amplitude', 'SF_ML_gamma',
            'IAR_phi',
            'LinearTrend',
        )

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        features_keys = self._feature_keys_for_new_feature_space()
        return features_keys

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'time', 'magnitude', 'band', 'error'

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(self, detections, band, **kwargs):
        """
        Compute features from turbo-fats.
        Parameters
        ----------
        detections : 
            Light curve from a single band and a single object.
        band : int
            Number of the band of the light curve.
        kwargs
        Returns
        ------
        pd.DataFrame
            turbo-fats features (one-row dataframe).
        """
        columns = self.get_features_keys_with_band(band)

        def aux_function(oid_detections, **kwargs):
            if band not in oid_detections['band'].values:
                oid = oid_detections.index.values[0]
                logging.debug(
                    f'extractor=TURBOFATS object={oid} required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)

            oid_band_detections = oid_detections[oid_detections['band'] == band]

            object_features = self.feature_space.calculate_features(
                oid_band_detections)
            if len(object_features) == 0:
                return self.nan_series_in_band(band)
            out = pd.Series(
                data=object_features.values.flatten(),
                index=columns)
            return out

        features = detections.apply(aux_function)
        features.index.rename("oid", inplace=True)
        return features
