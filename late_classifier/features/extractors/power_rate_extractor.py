from typing import List
import numpy as np
import pandas as pd
from ..core.base import FeatureExtractor
from ..extractors import PeriodExtractor
import numba
import logging


@numba.jit(nopython=True)
def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True


class PowerRateExtractor(FeatureExtractor):
    def __init__(self):
        self.factors = [0.25, 1/3, 0.5, 2.0, 3.0, 4.0]

    def get_features_keys(self) -> List[str]:
        return [
            'Power_rate_1/4',
            'Power_rate_1/3',
            'Power_rate_1/2',
            'Power_rate_2',
            'Power_rate_3',
            'Power_rate_4'
        ]

    def get_required_keys(self) -> List[str]:
        return [
            'mjd',
            'magpsf',
            'magpsf_corr',
            'fid',
            'sigmapsf_corr_ext',
            'sigmapsf']

    def _compute_features(self, detections: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if ('shared_data' in kwargs.keys() and
                'periodogram' in kwargs['shared_data'].keys()):
            periodograms = kwargs['shared_data']['periodogram']
        else:
            logging.info('PowerRateExtractor was not provided with periodogram '
                         'data, so a periodogram is being computed')
            shared_data = dict()
            period_extractor = PeriodExtractor()
            _ = period_extractor.compute_features(
                detections,
                shared_data=shared_data)
            periodograms = shared_data['periodogram']

        oids = detections.index.unique()
        power_rates = []
        for oid in oids:
            power_rate_values = []
            if oid in periodograms.keys():
                for factor in self.factors:
                    power_rate_values.append(
                        self._get_power_ratio(periodograms[oid], factor))
                power_rates.append(power_rate_values)
            else:
                power_rates.append(
                    [np.nan]*len(self.factors)
                )
        power_rates = pd.DataFrame(
            data=np.array(power_rates),
            columns=self.get_features_keys(),
            index=oids)
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
