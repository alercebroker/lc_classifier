from typing import Tuple
import numpy as np
import pandas as pd
from ..core.base import FeatureExtractor
from P4J import MultiBandPeriodogram
import logging
from functools import lru_cache
from typing import List, Optional


class PeriodExtractor(FeatureExtractor):
    def __init__(
            self, 
            bands: List[str], 
            smallest_period: float, 
            largest_period: float, 
            optimal_grid : bool,
            trim_lightcurve_to_n_days: Optional[float],
            min_length: int):
        
        self.optimal_grid = optimal_grid
        self.periodogram_computer = MultiBandPeriodogram(method='MHAOV')
        self.bands = bands
        self.smallest_period = smallest_period
        self.largest_period = largest_period
        self.trim_lightcurve_to_n_days = trim_lightcurve_to_n_days
        self.min_length = min_length

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

    def _trim_lightcurve(self, lightcurve: pd.DataFrame):
        if self.trim_lightcurve_to_n_days is None:
            return lightcurve
        
        times = lightcurve['time'].values

        best_starting = 0
        best_ending = 0

        starting = 0
        ending = 0
        while True:
            current_n = ending - starting
            if current_n > (best_ending - best_starting):
                best_starting = starting
                best_ending = ending
            current_timespan = times[ending] - times[starting]
            if current_timespan < self.trim_lightcurve_to_n_days:
                ending += 1
                if ending >= len(times):
                    break
            else:
                starting += 1

        return lightcurve.iloc[best_starting:(best_ending+1)]
    
    def _compute_features_from_df_groupby(
            self, detections, **kwargs) -> pd.DataFrame:
        def aux_function(oid_detections, **kwargs):
            oid = oid_detections.index.values[0]
            oid_detections = self._trim_lightcurve(oid_detections)
            oid_detections = oid_detections.groupby('band').filter(
                lambda x: len(x) > self.min_length).sort_values('time')
            
            if len(oid_detections) == 0:
                object_features = self.nan_series()
                kwargs['periodograms'][oid] = {
                    'freq': None,
                    'per': None
                }
                return object_features

            bands = oid_detections['band'].values
            available_bands = np.unique(bands)

            self.periodogram_computer.set_data(
                mjds=oid_detections['time'].values,
                mags=oid_detections['magnitude'].values,
                errs=oid_detections['error'].values,
                fids=bands)

            try:
                if self.optimal_grid:
                    self.periodogram_computer.optimal_frequency_grid_evaluation(
                        smallest_period=self.smallest_period,
                        largest_period=self.largest_period,
                        shift=0.2
                    )
                    self.frequencies = self.periodogram_computer.optimal_finetune_best_frequencies(
                        times_finer=10.0, n_local_optima=10)
                else:
                    self.periodogram_computer.frequency_grid_evaluation(
                        fmin=1/self.largest_period, 
                        fmax=1/self.smallest_period, 
                        fresolution=1e-3)
                    self.frequencies = self.periodogram_computer.finetune_best_frequencies(
                        n_local_optima=10, fresolution=1e-4)
                best_freq, best_per = self.periodogram_computer.get_best_frequencies()
                if len(best_freq) == 0:
                    logging.error(f'[PeriodExtractor] best frequencies has len 0: '
                                f'oid {oid}')
                    object_features = self.nan_series()
                    kwargs['periodograms'][oid] = {
                        'freq': None,
                        'per': None
                    }
                    return object_features
            except TypeError as e:
                logging.error(f'TypeError exception in PeriodExtractor: '
                              f'oid {oid}\n{e}')
                object_features = self.nan_series()
                kwargs['periodograms'][oid] = {
                    'freq': None,
                    'per': None
                }
                return object_features
            
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
