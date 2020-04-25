from typing import List

from late_classifier.features.core.base import FeatureExtractor
import pandas as pd
import logging


class ColorFeatureExtractor(FeatureExtractor):
    def get_features_keys(self) -> List[str]:
        return ['g-r_max', 'g-r_mean']

    def get_required_keys(self) -> List[str]:
        return ['fid', 'magpsf_corr']

    def _compute_features(self, detections, **kwargs):
        """
        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with detections of an object.

        kwargs Not required.

        Returns :class:pandas.`DataFrame`
        -------
        """
        # pd.options.display.precision = 10
        index = detections.index[0]
        if 1 not in detections.fid.unique() or 2 not in detections.fid.unique():
            logging.warning(f'extractor=COLOR  object={index}  required_cols={self.get_required_keys()}  filters_qty=2')
            return self.nan_df(index)

        g_band_mag = detections[detections.fid == 1]['magpsf_corr'].values
        r_band_mag = detections[detections.fid == 2]['magpsf_corr'].values

        g_r_max = g_band_mag.min() - r_band_mag.min()
        g_r_mean = g_band_mag.mean() - r_band_mag.mean()

        data = [g_r_max, g_r_mean]

        return pd.DataFrame([data], columns=self.get_features_keys(), index=[index])
