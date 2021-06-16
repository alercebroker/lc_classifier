from typing import Tuple
from functools import lru_cache
import logging
import numpy as np
import pandas as pd

from ..core.base import FeatureExtractorSingleBand
from ..extractors import PeriodExtractor


class FoldedKimExtractor(FeatureExtractorSingleBand):
    """https://github.com/dwkim78/upsilon/blob/
    64b2b7f6e3b3991a281182549959aabd385a6f28/
    upsilon/extract_features/extract_features.py#L151"""

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        return 'Psi_CS', 'Psi_eta'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'time', 'magnitude'

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)
    
    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs) -> pd.DataFrame:
        
        if ('shared_data' in kwargs.keys() and
                'period' in kwargs['shared_data'].keys()):
            periods = kwargs['shared_data']['period']
        else:
            logging.info('Folded Kim extractor was not provided with period '
                         'data, so a periodogram is being computed')
            period_extractor = PeriodExtractor(self.bands)
            periods = period_extractor.compute_features(detections)

        columns = self.get_features_keys_with_band(band)

        def aux_function(oid_detections, band, **kwargs):
            oid = oid_detections.index.values[0]
            if band not in oid_detections['band'].values:
                logging.debug(
                    f'extractor=Folded Kim extractor object={oid} '
                    f'required_cols={self.get_required_keys()}  band={band}')
                return self.nan_series_in_band(band)

            try:
                oid_period = periods[['Multiband_period']].loc[[oid]].values.flatten()
            except KeyError as e:
                logging.error(f'KeyError in FoldedKimExtractor, period is not '
                              f'available: oid {oid}\n{e}')
                return self.nan_series_in_band(band)

            oid_band_detections = oid_detections[oid_detections['band'] == band]
            lc_len = len(oid_band_detections)
            if lc_len <= 2:
                psi_cumsum = psi_eta = np.nan
            else:
                time = oid_band_detections['time'].values
                magnitude = oid_band_detections['magnitude'].values

                folded_time = np.mod(time, 2 * oid_period) / (2 * oid_period)
                sorted_mags = magnitude[np.argsort(folded_time)]
                sigma = np.std(sorted_mags)
                if sigma != 0.0:
                    m = np.mean(sorted_mags)
                    s = np.cumsum(sorted_mags - m) * 1.0 / (lc_len * sigma)
                    psi_cumsum = np.max(s) - np.min(s)
                    sigma_squared = sigma ** 2
                    psi_eta = (1.0 / ((lc_len - 1) * sigma_squared) *
                               np.sum(np.power(sorted_mags[1:] - sorted_mags[:-1], 2)))
                else:
                    psi_cumsum = psi_eta = np.nan

            out = pd.Series(
                data=[psi_cumsum, psi_eta],
                index=columns)
            return out
        
        features = detections.apply(lambda det: aux_function(det, band))
        features.index.name = 'oid'
        return features
