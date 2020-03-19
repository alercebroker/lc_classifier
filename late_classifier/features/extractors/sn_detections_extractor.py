from late_classifier.features.core.base import FeatureExtractorSingleBand
import pandas as pd
import numpy as np


class SupernovaeDetectionFeatureComputer(FeatureExtractorSingleBand):

    def __init__(self):
        super().__init__()
        self.features_keys = ['delta_mag_fid',
                              'delta_mjd_fid',
                              'first_mag',
                              'mean_mag',
                              'min_mag',
                              'n_det',
                              'n_neg',
                              'n_pos',
                              'positive_fraction']

    def _compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with single band detections of an object.

        kwargs Not required.

        Returns :class:pandas.`DataFrame`
        -------

        """
        if len(detections.index.unique()) > 1:
            raise Exception('SupernovaeDetectionFeatureComputer handles one lightcurve at a time')

        detections = detections.sort_values('mjd')
        count = len(detections)
        if count == 0:
            features = pd.DataFrame(
                columns=self.features_keys
            )
            features.index.name = 'oid'
            return features

        n_pos = len(detections[detections.isdiffpos > 0])
        n_neg = len(detections[detections.isdiffpos < 0])
        min_mag = detections['magpsf_corr'].min()
        first_mag = detections.iloc[0]['magpsf_corr']
        delta_mjd_fid = detections.iloc[count-1]['mjd'] - detections.iloc[0]['mjd']
        delta_mag_fid = detections['magpsf_corr'].max() - min_mag
        positive_fraction = n_pos/(n_pos + n_neg)
        mean_mag = detections['magpsf_corr'].mean()

        data = {
            'oid': detections.index[0],
            'delta_mag_fid': delta_mag_fid,
            'delta_mjd_fid': delta_mjd_fid,
            'first_mag': first_mag,
            'mean_mag': mean_mag,
            'min_mag': min_mag,
            'n_det': n_neg + n_pos,
            'n_neg': n_neg,
            'n_pos': n_pos,
            'positive_fraction': positive_fraction
        }
        return pd.DataFrame.from_records([data], index='oid')
