from late_classifier.features.core.base import FeatureExtractorSingleBand
from late_classifier.features.extractors.sn_detections_extractor import SupernovaeDetectionFeatureExtractor
import pandas as pd
import numpy as np
import logging


class SupernovaeNonDetectionFeatureExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        super().__init__()
        self.features_keys = ['dmag_first_det_fid',
                              'dmag_non_det_fid',
                              'last_diffmaglim_before_fid',
                              'last_mjd_before_fid',
                              'max_diffmaglim_before_fid',
                              'median_diffmaglim_before_fid',
                              'n_non_det_before_fid',
                              'n_non_det_after_fid',
                              'max_diffmaglim_after_fid',
                              'median_diffmaglim_after_fid'
                              ]
        self.required_keys = ["diffmaglim", "mjd", "fid"]

    def compute_before_features(self, det_result, non_detections, band):
        """

        Parameters
        ----------
        det_result :class:pandas.`DataFrame`
        DataFrame with Supernovae Detections Features.

        non_detections :class:pandas.`DataFrame`
        Dataframe with non detections before a specific mjd.

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

    def _compute_features(self, detections, band=None, **kwargs):
        """

        Parameters
        ----------
        detections class:pandas.`DataFrame`
        kwargs Required non_detections class:pandas.`DataFrame`

        Returns class:pandas.`DataFrame`
        -------

        """
        if len(detections.index.unique()) > 1:
            raise Exception('SupernovaeNonDetectionFeatureExtractor handles one lightcurve at a time')
        required = ['non_detections']
        for key in required:
            if key not in kwargs:
                raise Exception(f'SupernovaeNonDetectionFeatureExtractor required {key} argument')

        index = detections.index[0]
        detections = detections[detections.fid == band]
        non_detections = kwargs['non_detections'].sort_values('mjd')
        non_detections = non_detections[non_detections.fid == band]
        columns = self.get_features_keys(band)

        if band is None or len(detections) == 0:
            logging.error(
                f'Input dataframe invalid {index}\n - Required columns: {self.required_keys}\n - Required one filter.')
            nan_df = self.nan_df(index)
            nan_df.columns = columns
            return nan_df

        detections = detections.sort_values('mjd')
        first_mjd = detections["mjd"].iloc[0]
        fid = detections["fid"].iloc[0]

        non_detections_before = non_detections[non_detections.mjd < first_mjd]
        non_detections_after = non_detections[non_detections.mjd >= first_mjd]

        det_result = SupernovaeDetectionFeatureExtractor()._compute_features(detections, band=band)
        non_det_before = self.compute_before_features(det_result, non_detections_before, band)
        non_det_after = self.compute_after_features(non_detections_after)

        features = {
            "fid": fid,
            **det_result,
            **non_det_before,
            **non_det_after
        }
        df = pd.DataFrame.from_dict(features)
        df.index = [index]
        return df
