from typing import Tuple, List
from functools import lru_cache

from ..core.base import FeatureExtractorSingleBand
import pandas as pd
import numpy as np
import logging
import mhps


class MHPSExtractor(FeatureExtractorSingleBand):
    def __init__(self, bands: List, t1=100, t2=10, dt=3.0, mag0=19.0, epsilon=1.0):
        super().__init__(bands)
        self.t1 = t1
        self.t2 = t2
        self.dt = dt
        self.mag0 = mag0
        self.epsilon = epsilon

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        return (
            'MHPS_ratio',
            'MHPS_low',
            'MHPS_high',
            'MHPS_non_zero',
            'MHPS_PN_flag')

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'time', 'magnitude', 'error', 'band'

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs):
        columns = self.get_features_keys_with_band(band)

        def aux_function(oid_detections, band, **kwargs):
            if band not in oid_detections['band'].values:
                oid = oid_detections.index.values[0]
                logging.debug(
                    f'extractor=MHPS object={oid} required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)
            
            oid_band_detections = oid_detections[oid_detections['band'] == band].sort_values('time')

            mag = oid_band_detections['magnitude'].values.astype(np.double)
            magerr = oid_band_detections['error'].values.astype(np.double)
            time = oid_band_detections['time'].values.astype(np.double)
            ratio, low, high, non_zero, pn_flag = mhps.statistics(
                mag,
                magerr,
                time,
                self.t1,
                self.t2)
            return pd.Series(
                data=[ratio, low, high, non_zero, pn_flag],
                index=columns)
        
        mhps_results = detections.apply(lambda det: aux_function(det, band))
        mhps_results.index.name = 'oid'
        return mhps_results
