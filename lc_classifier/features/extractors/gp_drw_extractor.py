from typing import List

import numpy as np
import pandas as pd
import logging

from ..core.base import FeatureExtractorSingleBand

import celerite2
from celerite2 import terms

from scipy.optimize import minimize


class GPDRWExtractor(FeatureExtractorSingleBand):
    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs) -> pd.DataFrame:
        def aux_function(oid_detections, **kwargs):
            if band not in oid_detections.fid.values:
                logging.info(
                    "extractor=GP DRW extractor object=%s required_cols=%s band=%s",
                    oid_detections.index.values[0],
                    self.get_required_keys(),
                    band)
                return self.nan_series_in_band(band)

            oid_band_detections = oid_detections[oid_detections.fid == band]
            lc = oid_band_detections[[
                'mjd', 'magpsf_ml', 'sigmapsf_ml']].values
            time = lc[:, 0]
            mag = lc[:, 1]
            err = lc[:, 2]

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

            initial_params = [np.log(1.0), np.log(1.0)]
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

        features = detections.apply(aux_function)
        features.index.name = 'oid'
        return features

    def get_features_keys(self) -> List[str]:
        feature_names = [
            'GP_DRW_sigma',
            'GP_DRW_tau'
        ]
        return feature_names

    def get_required_keys(self) -> List[str]:
        return [
            'mjd',
            'magpsf_ml',
            'fid',
            'sigmapsf_ml'
        ]
