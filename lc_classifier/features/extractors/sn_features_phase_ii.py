from typing import Tuple, List
import numpy as np
import pandas as pd
from functools import lru_cache
import logging

from ..core.base import FeatureExtractorSingleBand
from ...utils import is_sorted, first_occurrence


class SNFeaturesPhaseIIExtractor(FeatureExtractorSingleBand):
    def __init__(self, bands: List):
        super().__init__(bands)
        logging.warning("This class is not fully implemented")  # TODO: finish implementation

    def compute_feature_in_one_band(
            self, detections: pd.DataFrame, band, **kwargs) -> pd.DataFrame:
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(
            grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs):

        def aux_function(oid_detections, **kwargs):
            bands = oid_detections['band'].values
            if band not in bands:
                oid = oid_detections.index.values[0]
                logging.debug(
                    f'extractor=SN features phase II'
                    f' object={oid} '
                    f'required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)

            band_mask = bands == band
            oid_band_detections = oid_detections[band_mask]

            diff_flux_band = oid_band_detections['difference_flux'].values

            # positive_fraction
            positive_observations = diff_flux_band > 0
            positive_fraction = positive_observations.astype(np.float).mean()

            # first detection in any band
            if not is_sorted(oid_detections['time'].values):
                oid_detections = oid_detections.sort_values('time')

            diff_flux_any_band = oid_detections['difference_flux'].values
            diff_err_any_band = oid_detections['difference_flux_error'].values

            # This value "3" is arbitrary
            # TODO: replace with a less arbitrary version
            detected = np.abs(diff_flux_any_band) - 3 * diff_err_any_band > 0
            first_detection_index = first_occurrence(detected, True)

            # no detections in any band
            if first_detection_index == -1:
                n_non_det_before_band = 0
                n_non_det_after_band = 0
                # frac_non_det_after_band = 0.0
                dflux_first_det_band = np.nan
                dflux_non_det_band = np.nan
                last_flux_before_band = np.nan
                max_flux_before_band = np.nan
                max_flux_after_band = np.nan
                median_flux_before_band = np.nan
                median_flux_after_band = np.nan

            else:
                # number of non detections in band before the 1st detection is any band
                n_non_det_before_band = (
                        band_mask[:first_detection_index]).astype(np.int).sum()

                # fraction of non detections in band after the 1st detection in any band
                n_det_after_band = (band_mask & detected)[first_detection_index:].astype(np.float).sum()
                n_non_det_after_band = (band_mask & ~detected)[first_detection_index:].astype(np.float).sum()

                # frac_non_det_after_band = n_det_after_band / (n_det_after_band + n_non_det_after_band)

                # diff of flux between first det in any band and previous non det in band
                flux_1st_det = diff_flux_any_band[first_detection_index]
                before_1st_det = oid_detections.iloc[:first_detection_index]
                before_1st_det_in_band = before_1st_det[before_1st_det['band'] == band]
                if len(before_1st_det_in_band) == 0:
                    dflux_first_det_band = np.nan
                    dflux_non_det_band = np.nan
                    last_flux_before_band = np.nan
                    max_flux_before_band = np.nan
                    median_flux_before_band = np.nan
                else:
                    non_det_flux_in_band = before_1st_det_in_band['difference_flux'].values
                    dflux_first_det_band = flux_1st_det - non_det_flux_in_band[-1]
                    dflux_non_det_band = flux_1st_det - np.median(non_det_flux_in_band)
                    last_flux_before_band = non_det_flux_in_band[-1]
                    max_flux_before_band = np.max(non_det_flux_in_band)
                    median_flux_before_band = np.median(non_det_flux_in_band)

                after_mask = np.zeros(len(bands), dtype=np.bool)
                after_mask[first_detection_index:] = True
                non_det_after_band = diff_flux_any_band[~detected & band_mask & after_mask]
                if len(non_det_after_band) == 0:
                    max_flux_after_band = np.nan
                    median_flux_after_band = np.nan
                else:
                    max_flux_after_band = np.max(non_det_after_band)
                    median_flux_after_band = np.median(non_det_after_band)

            data = [
                positive_fraction,
                dflux_first_det_band,
                dflux_non_det_band,
                last_flux_before_band,
                max_flux_before_band,
                max_flux_after_band,
                median_flux_before_band,
                median_flux_after_band,
                n_non_det_before_band,
                n_non_det_after_band,
                # frac_non_det_after_band
            ]
            features_series = pd.Series(
                data=data,
                index=self.get_features_keys_with_band(band))
            return features_series

        features = detections.apply(aux_function)
        features.index.name = 'oid'
        return features

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        return (
            'positive_fraction',

            'dflux_first_det_band',
            'dflux_non_det_band',
            'last_flux_before_band',
            'max_flux_before_band',
            'max_flux_after_band',
            'median_flux_before_band',
            'median_flux_after_band',
            'n_non_det_before_band',
            'n_non_det_after_band',
            # 'frac_non_det_after_band'
        )

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return "difference_flux", "difference_flux_error", "time", "band"
