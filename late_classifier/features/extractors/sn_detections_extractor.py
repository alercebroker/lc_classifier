from typing import List

from late_classifier.features.core.base import FeatureExtractorSingleBand
import pandas as pd
import logging


class SupernovaeDetectionFeatureExtractor(FeatureExtractorSingleBand):
    def get_features_keys(self) -> List[str]:
        return ['delta_mag_fid',
                'delta_mjd_fid',
                'first_mag',
                'mean_mag',
                'min_mag',
                'n_det',
                'n_neg',
                'n_pos',
                'positive_fraction']

    def get_required_keys(self) -> List[str]:
        return ["isdiffpos", "magpsf_corr", "mjd"]

    def compute_feature_in_one_band(self, detections, band=None, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with single band detections of an object.

        band :class:int

        kwargs Not required.

        Returns :class:pandas.`DataFrame`
        -------

        """

        index = detections.index.unique()[0]
        columns = self.get_features_keys_with_band(band)
        detections = detections[detections.fid == band]

        if band is None or len(detections) == 0:
            logging.warning(f'extractor=SNDET  object={index}  required_cols={self.get_required_keys()} filters_qty=1')
            nan_df = self.nan_df(index)
            nan_df.columns = columns
            return nan_df

        detections = detections.sort_values('mjd')

        n_pos = len(detections[detections.isdiffpos > 0])
        n_neg = len(detections[detections.isdiffpos < 0])
        min_mag = detections['magpsf_corr'].values.min()
        first_mag = detections['magpsf_corr'].values[0]
        delta_mjd_fid = detections['mjd'].values[-1] - detections['mjd'].values[0]
        delta_mag_fid = detections['magpsf_corr'].values.max() - min_mag
        positive_fraction = n_pos/(n_pos + n_neg)
        mean_mag = detections['magpsf_corr'].values.mean()

        data = [delta_mag_fid,
                delta_mjd_fid,
                first_mag,
                mean_mag,
                min_mag,
                n_neg + n_pos,
                n_neg,
                n_pos,
                positive_fraction]
        return pd.DataFrame.from_records([data], columns=columns, index=[index])
