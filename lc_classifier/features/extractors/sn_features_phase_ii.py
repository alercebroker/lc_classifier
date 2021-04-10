from typing import Tuple
import numpy as np
import pandas as pd
from functools import lru_cache
import logging

from ..core.base import FeatureExtractorSingleBand
from .utils import is_sorted, first_occurrence


class SNFeaturesPhaseIIExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        super(SNFeaturesPhaseIIExtractor, self).__init__()
        logging.warning("This class is not fully implemented")  # TODO: finish implementation

    def compute_feature_in_one_band(
            self, detections: pd.DataFrame, band, **kwargs) -> pd.DataFrame:
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(
            grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs):

        def aux_function(oid_detections, **kwargs):
            fids = oid_detections['fid'].values
            if band not in fids:
                oid = oid_detections.index.values[0]
                logging.debug(
                    f'extractor=SN features phase II'
                    f' object={oid} '
                    f'required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)

            band_mask = fids == band
            oid_band_detections = oid_detections[band_mask]

            diff_flux_band = oid_band_detections['diff_flux'].values

            mag_band = oid_band_detections['magpsf_ml'].dropna().values

            # delta_mag_fid
            if len(mag_band) > 0:
                delta_mag_fid = mag_band.max() - mag_band.min()
            else:
                delta_mag_fid = np.nan

            # positive_fraction
            positive_observations = diff_flux_band > 0
            positive_fraction = positive_observations.astype(np.float).mean()

            # TODO: dflux_first_det_fid

            # first detection in any band
            if not is_sorted(oid_detections['mjd'].values):
                oid_detections = oid_detections.sort_values('mjd')

            diff_flux_any_band = oid_detections['diff_flux'].values
            diff_err_any_band = oid_detections['diff_err'].values

            # This value "3" is arbitrary
            detected = np.abs(diff_flux_any_band) - 3 * diff_err_any_band > 0
            first_detection_index = first_occurrence(detected, True)
            if first_detection_index == -1:
                n_non_det_before_fid = 0
                frac_non_det_after_fid = 0.0

            else:
                # number of non detections in band before the 1st detection is any band
                n_non_det_before_fid = (
                        band_mask[:first_detection_index]).astype(np.int).sum()

                # fraction of non detections in band after the 1st detection in any band
                n_det_after_fid = (band_mask & detected)[first_detection_index:].astype(np.float).sum()
                n_non_det_after_fid = (band_mask & ~detected)[first_detection_index:].astype(np.float).sum()

                frac_non_det_after_fid = n_det_after_fid / (n_det_after_fid + n_non_det_after_fid)

            data = [
                delta_mag_fid,
                positive_fraction,
                n_non_det_before_fid,
                frac_non_det_after_fid
            ]
            features_series = pd.Series(
                data=data,
                index=self.get_features_keys_with_band(band))
            return features_series

        features = detections.apply(aux_function)
        features.index.name = 'oid'
        return features

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return (
            'delta_mag_fid',
            'positive_fraction',

            #'dflux_first_det_fid',
            #'dmag_non_det_fid',
            'n_non_det_before_fid',
            'frac_non_det_after_fid'
        )

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return "diff_flux", "diff_err", "mjd", "fid", "magpsf_ml"
