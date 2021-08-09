from typing import Tuple, List
from functools import lru_cache

from ..core.base import FeatureExtractorSingleBand
from .sn_detections_extractor import SupernovaeDetectionFeatureExtractor
from ...utils import is_sorted
import pandas as pd
import numpy as np
import logging


class SupernovaeDetectionAndNonDetectionFeatureExtractor(FeatureExtractorSingleBand):
    def __init__(self, bands: List):
        super().__init__(bands)
        self.supernovae_detection_extractor = SupernovaeDetectionFeatureExtractor(bands)

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        return (
            'delta_mag_fid',
            'delta_mjd_fid',
            'first_mag',
            'mean_mag',
            'min_mag',
            'n_det',
            'n_neg',
            'n_pos',
            'positive_fraction',
            'n_non_det_before_fid',
            'max_diffmaglim_before_fid',
            'median_diffmaglim_before_fid',
            'last_diffmaglim_before_fid',
            'last_mjd_before_fid',
            'dmag_non_det_fid',
            'dmag_first_det_fid',
            'n_non_det_after_fid',
            'max_diffmaglim_after_fid',
            'median_diffmaglim_after_fid'
        )

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return self.supernovae_detection_extractor.get_required_keys()

    @lru_cache(1)
    def get_non_detections_required_keys(self):
        return "diffmaglim", "time", "band"

    def compute_before_features(self, det_result, non_detections, band):
        """

        Parameters
        ----------
        det_result :class:pandas.`DataFrame`
        DataFrame with Supernovae Detections Features.

        non_detections :class:pandas.`DataFrame`
        Dataframe with non detections before a specific mjd.

        band :class:int

        Returns class:pandas.`DataFrame`
        -------

        """
        count = len(non_detections)
        if count == 0:
            max_diffmaglim_before_fid = np.nan
            median_diffmaglim_before_fid = np.nan
            last_diffmaglim_before_fid = np.nan
            last_mjd_before_fid = np.nan
            dmag_non_det_fid = np.nan
            dmag_first_det_fid = np.nan
        else:
            non_detections_diffmaglim = non_detections['diffmaglim']
            max_diffmaglim_before_fid = non_detections_diffmaglim.max()
            median_diffmaglim_before_fid = non_detections_diffmaglim.median()
            last_diffmaglim_before_fid = non_detections.iloc[-1]['diffmaglim']
            last_mjd_before_fid = non_detections.iloc[-1]['time']
            dmag_non_det_fid = median_diffmaglim_before_fid - \
                det_result[f'min_mag_{band}'].iloc[0]
            dmag_first_det_fid = last_diffmaglim_before_fid - \
                det_result[f'first_mag_{band}'].iloc[0]

        feature_names = [
            'n_non_det_before_fid',
            'max_diffmaglim_before_fid',
            'median_diffmaglim_before_fid',
            'last_diffmaglim_before_fid',
            'last_mjd_before_fid',
            'dmag_non_det_fid',
            'dmag_first_det_fid'
        ]
        feature_names = [n+f'_{band}' for n in feature_names]

        values = [
            count,
            max_diffmaglim_before_fid,
            median_diffmaglim_before_fid,
            last_diffmaglim_before_fid,
            last_mjd_before_fid,
            dmag_non_det_fid,
            dmag_first_det_fid
        ]

        series = pd.Series(values, index=feature_names)
        return series

    def compute_after_features(self, non_detections, band):
        """

        Parameters
        ----------
        non_detections :class:pandas.`DataFrame`
        Dataframe with non detections after a specific mjd.

        Returns class:pandas.`DataFrame`
        -------

        """
        count = len(non_detections)
        non_detections_diffmaglim = non_detections['diffmaglim']

        feature_names = [
            'n_non_det_after_fid',
            'max_diffmaglim_after_fid',
            'median_diffmaglim_after_fid'
        ]
        feature_names = [n+f'_{band}' for n in feature_names]

        values = [
            count,
            np.nan if count == 0 else non_detections_diffmaglim.max(),
            np.nan if count == 0 else non_detections_diffmaglim.median()
        ]

        series = pd.Series(values, index=feature_names)
        return series

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(self, detections, band, **kwargs):
        """

        Parameters
        ----------
        detections
        kwargs Required non_detections class:pandas.`DataFrame`

        Returns class:pandas.`DataFrame`
        -------

        """
        required = ['non_detections']
        for key in required:
            if key not in kwargs:
                raise Exception(
                    f'SupernovaeDetectionAndNonDetectionFeatureExtractor requires {key} argument')
        
        non_detections = kwargs['non_detections']
        non_detection_keys = set(non_detections.columns)
        required_non_detection_keys = set(
            self.get_non_detections_required_keys())
        non_det_key_intersection = non_detection_keys.intersection(
            required_non_detection_keys)

        if len(non_det_key_intersection) != len(required_non_detection_keys):
            raise Exception(
                "Non detections dataframe does not have all required columns")

        non_det_unique_oids = non_detections.index.unique()

        def aux_function(oid_detections, **kwargs):
            oid = oid_detections.index.values[0]
            if band not in oid_detections['band'].values:
                logging.debug(
                    f'extractor=SN detection object={oid} required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)
            
            oid_band_detections = oid_detections[oid_detections['band'] == band]

            first_mjd = oid_band_detections["time"].min()

            if oid not in non_det_unique_oids:
                oid_non_detections = pd.DataFrame(
                    columns=non_detections.columns)
            else:
                oid_non_detections = non_detections.loc[[oid]]

            oid_band_non_detections = oid_non_detections[oid_non_detections['band'] == band]
            if (len(oid_band_non_detections) != 0 and
                not is_sorted(oid_band_non_detections['time'].values)):
                oid_band_non_detections = oid_band_non_detections.sort_values('time')

            before_mask = oid_band_non_detections['time'] < first_mjd
            non_detections_before = oid_band_non_detections[before_mask]
            non_detections_after = oid_band_non_detections[~before_mask]

            det_result = self.supernovae_detection_extractor.compute_feature_in_one_band(
                oid_band_detections, band)
            non_det_before = self.compute_before_features(
                det_result, non_detections_before, band)
            non_det_after = self.compute_after_features(non_detections_after, band)

            series = pd.concat((det_result.iloc[0], non_det_before, non_det_after))
            series.name = oid
            return series

        sn_features = detections.apply(aux_function)
        sn_features.index.name = 'oid'
        sn_features = sn_features.astype(float)
        return sn_features
