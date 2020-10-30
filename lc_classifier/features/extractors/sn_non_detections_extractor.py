from typing import List

from ..core.base import FeatureExtractorSingleBand
from .sn_detections_extractor import SupernovaeDetectionFeatureExtractor
import pandas as pd
import numpy as np
import logging


class SupernovaeDetectionAndNonDetectionFeatureExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        self.supernovae_detection_extractor = SupernovaeDetectionFeatureExtractor()

    def get_features_keys(self) -> List[str]:
        return ['delta_mag_fid',
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
                'median_diffmaglim_after_fid']

    def get_required_keys(self) -> List[str]:
        return self.supernovae_detection_extractor.get_required_keys()

    def get_non_detections_required_keys(self):
        return ["diffmaglim", "mjd", "fid"]

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
        count = len(non_detections['fid'])
        return {
            'n_non_det_before_fid': count,
            'max_diffmaglim_before_fid': np.nan if count == 0 else non_detections['diffmaglim'].max(),
            'median_diffmaglim_before_fid': np.nan if count == 0 else non_detections['diffmaglim'].median(),
            'last_diffmaglim_before_fid': np.nan if count == 0 else non_detections.iloc[-1].diffmaglim,
            'last_mjd_before_fid': np.nan if count == 0 else non_detections.iloc[-1].mjd,
            'dmag_non_det_fid': np.nan if count == 0 else non_detections['diffmaglim'].median() - det_result[f'min_mag_{band}'],
            'dmag_first_det_fid': np.nan if count == 0 else non_detections.iloc[-1].diffmaglim - det_result[f'first_mag_{band}']
        }

    def compute_after_features(self, non_detections):
        """

        Parameters
        ----------
        non_detections :class:pandas.`DataFrame`
        Dataframe with non detections after a specific mjd.

        Returns class:pandas.`DataFrame`
        -------

        """
        count = len(non_detections['fid'])
        return {
            'n_non_det_after_fid': count,
            'max_diffmaglim_after_fid': np.nan if count == 0 else non_detections['diffmaglim'].max(),
            'median_diffmaglim_after_fid': np.nan if count == 0 else non_detections['diffmaglim'].median()
        }

    def compute_feature_in_one_band(self, detections, band=None, **kwargs):
        """

        Parameters
        ----------
        detections class:pandas.`DataFrame`
        kwargs Required non_detections class:pandas.`DataFrame`

        Returns class:pandas.`DataFrame`
        -------

        """
        required = ['non_detections']
        for key in required:
            if key not in kwargs:
                raise Exception(f'SupernovaeDetectionAndNonDetectionFeatureExtractor requires {key} argument')

        non_detections = kwargs['non_detections'].sort_values('mjd')

        non_detection_keys = set(non_detections.columns)
        required_non_detection_keys = set(self.get_non_detections_required_keys())
        non_det_key_intersection = non_detection_keys.intersection(required_non_detection_keys)

        if len(non_det_key_intersection) != len(required_non_detection_keys):
            raise Exception
        oids = detections.index.unique()
        sn_features = []

        detections = detections.sort_values('mjd')
        columns = self.get_features_keys_with_band(band)

        for oid in oids:
            oid_detections = detections.loc[[oid]]
            if band not in oid_detections.fid.values:
                logging.debug(
                    f'extractor=SN detection object={oid} required_cols={self.get_required_keys()} band={band}')
                nan_df = self.nan_df(oid)
                nan_df.columns = columns
                sn_features.append(nan_df)
                continue

            oid_band_detections = oid_detections[oid_detections.fid == band]

            first_mjd = oid_band_detections["mjd"].iloc[0]

            if oid not in non_detections.index.unique():
                oid_non_detections = pd.DataFrame(columns=non_detections.columns)
            else:
                oid_non_detections = non_detections.loc[[oid]]

            oid_band_non_detections = oid_non_detections[oid_non_detections.fid == band]

            non_detections_before = oid_band_non_detections[oid_band_non_detections.mjd < first_mjd]
            non_detections_after = oid_band_non_detections[oid_band_non_detections.mjd >= first_mjd]

            det_result = self.supernovae_detection_extractor.compute_feature_in_one_band(
                oid_band_detections, band=band)
            non_det_before = self.compute_before_features(det_result, non_detections_before, band)
            non_det_after = self.compute_after_features(non_detections_after)

            features = {
                **det_result,
                **non_det_before,
                **non_det_after
            }
            df = pd.DataFrame.from_dict(features)
            df.index = [oid]
            df.columns = columns
            sn_features.append(df)
        sn_features = pd.concat(sn_features, axis=0)
        sn_features.index.name = 'oid'
        return sn_features
