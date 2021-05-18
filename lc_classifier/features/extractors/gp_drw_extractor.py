from typing import Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
import logging

from ..core.base import FeatureExtractorSingleBand
from ...utils import is_sorted

import celerite2
from celerite2 import terms

from scipy.optimize import minimize


class GPDRWExtractor(FeatureExtractorSingleBand):
    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs) -> pd.DataFrame:
        def aux_function(oid_detections, band, **kwargs):
            bands = oid_detections['band'].values
            if band not in bands:
                logging.info(
                    "extractor=GP DRW extractor object=%s required_cols=%s band=%s",
                    oid_detections.index.values[0],
                    self.get_required_keys(),
                    band)
                return self.nan_series_in_band(band)

            oid_band_detections = oid_detections[bands == band]

            if not is_sorted(oid_band_detections['time'].values):
                oid_band_detections.sort_values('time', inplace=True)

            time = oid_band_detections['time'].values
            mag = oid_band_detections['magnitude'].values
            err = oid_band_detections['error'].values

            time = time - time.min()
            mag = mag - mag.mean()
            sq_error = err ** 2

            kernel = terms.RealTerm(a=1.0, c=10.0)
            gp = celerite2.GaussianProcess(kernel, mean=0.0)

            def set_params(params, gp, time, sq_error):
                gp.mean = 0.0
                theta = np.exp(params)
                gp.kernel = terms.RealTerm(a=theta[0], c=theta[1])
                gp.compute(time, diag=sq_error, quiet=True)
                return gp

            def neg_log_like(params, gp, time, mag, sq_error):
                gp = set_params(params, gp, time, sq_error)
                return -gp.log_likelihood(mag)

            initial_params = np.zeros((2,), dtype=np.float)
            sol = minimize(
                neg_log_like,
                initial_params,
                method="L-BFGS-B",
                args=(gp, time, mag, sq_error))

            optimal_params = np.exp(sol.x)
            out_data = [
                optimal_params[0],
                1.0 / optimal_params[1]
            ]
            out = pd.Series(
                data=out_data, index=self.get_features_keys_with_band(band))
            return out

        features = detections.apply(lambda det: aux_function(det, band))
        features.index.name = 'oid'
        return features

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        feature_names = 'GP_DRW_sigma', 'GP_DRW_tau'
        return feature_names

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return (
            'time',
            'magnitude',
            'band',
            'error'
        )
