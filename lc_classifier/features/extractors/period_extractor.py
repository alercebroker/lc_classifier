from typing import Tuple
import numpy as np
import pandas as pd
from ..core.base import FeatureExtractor
from P4J import MultiBandPeriodogram
import logging
from functools import lru_cache


class PeriodExtractor(FeatureExtractor):
    def __init__(self, bands):
        self.periodogram_computer = MultiBandPeriodogram(method='MHAOV')
        self.bands = bands

    @lru_cache(maxsize=1)
    def get_features_keys(self) -> Tuple[str, ...]:
        features = ['Multiband_period', 'PPE']
        for band in self.bands:
            features.append(f'Period_band_{band}')
            features.append(f'delta_period_{band}')
        return tuple(features)

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return (
            'time',
            'magnitude',
            'error',
            'band'
        )

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)
    
    def _compute_features_from_df_groupby(
            self, detections, **kwargs) -> pd.DataFrame:
        def aux_function(oid_detections, **kwargs):
            oid = oid_detections.index.values[0]
            oid_detections = oid_detections.groupby('band').filter(
                lambda x: len(x) > 5).sort_values('time')

            bands = oid_detections['band'].values
            available_bands = np.unique(bands)

            self.periodogram_computer.set_data(
                mjds=oid_detections['time'].values,
                mags=oid_detections['magnitude'].values,
                errs=oid_detections['error'].values,
                fids=bands)

            try:
                self.periodogram_computer.frequency_grid_evaluation(
                    fmin=1e-3, fmax=20.0, fresolution=1e-3)
                self.frequencies = self.periodogram_computer.finetune_best_frequencies(
                    n_local_optima=10, fresolution=1e-4)
            except TypeError as e:
                logging.error(f'TypeError exception in PeriodExtractor: '
                              f'oid {oid}\n{e}')
                object_features = self.nan_series()
                kwargs['periodograms'][oid] = {
                    'freq': None,
                    'per': None
                }
                return object_features
            
            best_freq, best_per = self.periodogram_computer.get_best_frequencies()

            freq, per = self.periodogram_computer.get_periodogram()
            period_candidate = 1.0 / best_freq[0]

            period_candidates_per_band = []
            for band in self.bands:
                if band not in available_bands:
                    period_candidates_per_band.extend([np.nan, np.nan])
                    continue
                best_freq_band = self.periodogram_computer.get_best_frequency(band)

                # Getting best period
                best_period_band = 1.0 / best_freq_band

                # Calculating delta period
                delta_period_band = np.abs(period_candidate - best_period_band)
                period_candidates_per_band.extend([best_period_band, delta_period_band])

            # Significance estimation
            entropy_best_n = 100
            top_values = np.sort(per)[-entropy_best_n:]
            normalized_top_values = top_values + 1e-2
            normalized_top_values = normalized_top_values / np.sum(normalized_top_values)
            entropy = (-normalized_top_values * np.log(normalized_top_values)).sum()
            significance = 1 - entropy / np.log(entropy_best_n)
            object_features = pd.Series(
                data=[period_candidate, significance] + period_candidates_per_band,
                index=self.get_features_keys()
            )
            kwargs['periodograms'][oid] = {
                'freq': freq,
                'per': per
            }
            return object_features

        periodograms = {}
        features = detections.apply(aux_function, **{'periodograms': periodograms})
        features.index.name = 'oid'

        if 'shared_data' in kwargs.keys():
            kwargs['shared_data']['period'] = features
            kwargs['shared_data']['periodogram'] = periodograms
        return features
