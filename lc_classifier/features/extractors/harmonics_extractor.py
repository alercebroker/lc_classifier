from typing import Tuple, List

import numpy as np
import pandas as pd
import logging
from functools import lru_cache

from ..core.base import FeatureExtractorSingleBand
from ..extractors import PeriodExtractor


class HarmonicsExtractor(FeatureExtractorSingleBand):
    def __init__(self, bands: List):
        super().__init__(bands)
        self.n_harmonics = 7

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs) -> pd.DataFrame:

        if ('shared_data' in kwargs.keys() and
                'period' in kwargs['shared_data'].keys()):
            periods = kwargs['shared_data']['period']
        else:
            logging.info('Harmonics extractor was not provided with period '
                         'data, so a periodogram is being computed')
            period_extractor = PeriodExtractor(self.bands)
            periods = period_extractor.compute_features(detections)

        columns = self.get_features_keys_with_band(band)

        def aux_function(oid_detections, band, **kwargs):
            oid = oid_detections.index.values[0]
            if band not in oid_detections['band'].values:
                logging.debug(
                    f'extractor=Harmonics extractor object={oid} '
                    f'required_cols={self.get_required_keys()}  band={band}')
                return self.nan_series_in_band(band)
            
            oid_band_detections = oid_detections[oid_detections['band'] == band]

            magnitude = oid_band_detections['magnitude'].values
            time = oid_band_detections['time'].values
            error = oid_band_detections['error'].values + 10 ** -2

            try:
                period = periods[['Multiband_period']].loc[[oid]].values.flatten()
                best_freq = 1 / period

                omega = [np.array([[1.] * len(time)])]
                timefreq = (2.0 * np.pi * best_freq
                            * np.arange(1, self.n_harmonics + 1)).reshape(1, -1).T * time
                omega.append(np.cos(timefreq))
                omega.append(np.sin(timefreq))

                # Omega.shape == (lc_length, 1+2*self.n_harmonics)
                omega = np.concatenate(omega, axis=0).T
                
                inverr = 1.0 / error

                # weighted regularized linear regression
                w_a = inverr.reshape(-1, 1) * omega
                w_b = (magnitude * inverr).reshape(-1, 1)
                coeffs = np.matmul(np.linalg.pinv(w_a), w_b).flatten()
                fitted_magnitude = np.dot(omega, coeffs)
                coef_cos = coeffs[1:self.n_harmonics + 1]
                coef_sin = coeffs[self.n_harmonics + 1:]
                coef_mag = np.sqrt(coef_cos ** 2 + coef_sin ** 2)
                coef_phi = np.arctan2(coef_sin, coef_cos)

                # Relative phase
                coef_phi = coef_phi - coef_phi[0] * np.arange(1, self.n_harmonics + 1)
                coef_phi = coef_phi[1:] % (2 * np.pi)

                mse = np.mean((fitted_magnitude - magnitude) ** 2)
                out = pd.Series(
                    data=np.concatenate([coef_mag, coef_phi, np.array([mse])]),
                    index=columns)
                return out

            except Exception as e:
                logging.error(f'KeyError in HarmonicsExtractor, period is not '
                              f'available: oid {oid}\n{e}')
                return self.nan_series_in_band(band)
            
        features = detections.apply(lambda det: aux_function(det, band))
        features.index.name = 'oid'
        return features

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        feature_names = ['Harmonics_mag_%d' % (i+1) for i in range(self.n_harmonics)]
        feature_names += ['Harmonics_phase_%d' % (i+1) for i in range(1, self.n_harmonics)]
        feature_names.append('Harmonics_mse')
        return tuple(feature_names)

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return (
            'time',
            'magnitude',
            'band',
            'error'
        )
