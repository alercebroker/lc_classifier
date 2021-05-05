from typing import Tuple, List
from functools import lru_cache
import numpy as np
import pandas as pd
from ..core.base import FeatureExtractor
from ..extractors import PeriodExtractor
from ...utils import is_sorted
import logging


class PowerRateExtractor(FeatureExtractor):
    def __init__(self, bands: List):
        self.bands = bands
        self.factors = [0.25, 1/3, 0.5, 2.0, 3.0, 4.0]

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return (
            'Power_rate_1/4',
            'Power_rate_1/3',
            'Power_rate_1/2',
            'Power_rate_2',
            'Power_rate_3',
            'Power_rate_4'
        )

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return PeriodExtractor(self.bands).get_required_keys()

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)
    
    def _compute_features_from_df_groupby(self, detections, **kwargs) -> pd.DataFrame:
        if ('shared_data' in kwargs.keys() and
                'periodogram' in kwargs['shared_data'].keys()):
            periodograms = kwargs['shared_data']['periodogram']
        else:
            logging.info('PowerRateExtractor was not provided with periodogram '
                         'data, so a periodogram is being computed')
            shared_data = dict()
            period_extractor = PeriodExtractor(self.bands)
            _ = period_extractor.compute_features(
                detections,
                shared_data=shared_data)
            periodograms = shared_data['periodogram']

        def aux_function(oid_detections, **kwargs):
            oid = oid_detections.index.values[0]
            if oid in periodograms.keys() and periodograms[oid]['freq'] is not None:
                power_rate_values = []
                for factor in self.factors:
                    power_rate_values.append(
                        self._get_power_ratio(periodograms[oid], factor))
                return pd.Series(
                    data=power_rate_values,
                    index=self.get_features_keys())
            else:
                logging.error(f'PeriodPowerRateExtractor: period is not '
                              f'available for {oid}')
                return self.nan_series()

        power_rates = detections.apply(aux_function)
        power_rates.index.name = 'oid'
        return power_rates

    def _get_power_ratio(self, periodogram, period_factor):
        max_index = periodogram['per'].argmax()
        period = 1.0 / periodogram['freq'][max_index]
        period_power = periodogram['per'][max_index]

        freq = periodogram['freq']
        per = periodogram['per']
        if not is_sorted(freq):
            order = freq.argsort()
            freq = freq[order]
            per = per[order]
        desired_period = period * period_factor
        desired_frequency = 1.0 / desired_period
        i = np.searchsorted(freq, desired_frequency)

        if i == 0:
            i = 0
        elif desired_frequency > freq[-1]:
            i = len(freq) - 1
        else:
            left = freq[i-1]
            right = freq[i]
            mean = (left + right) / 2.0
            if desired_frequency > mean:
                i = i
            else:
                i = i-1
        return per[i] / period_power
