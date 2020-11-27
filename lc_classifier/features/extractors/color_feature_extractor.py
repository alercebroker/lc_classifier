from typing import List

from ..core.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging


class ColorFeatureExtractor(FeatureExtractor):
    def get_features_keys(self) -> List[str]:
        return ['g-r_max', 'g-r_mean', 'g-r_max_corr', 'g-r_mean_corr']

    def get_required_keys(self) -> List[str]:
        return ['fid', 'magpsf', 'magpsf_ml']

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        """
        Parameters
        ----------
        detections 
        DataFrame with detections of an object.
        kwargs Not required.
        Returns :class:pandas.`DataFrame`
        -------
        """
        # pd.options.display.precision = 10
        def aux_function(oid_detections):
            oid = oid_detections.index.values[0]
            if 1 not in oid_detections.fid.unique() or 2 not in oid_detections.fid.unique():
                logging.debug(
                    f'extractor=COLOR  object={oid}  required_cols={self.get_required_keys()}  filters_qty=2')
                return self.nan_series()

            g_band_mag_corr = oid_detections[oid_detections.fid == 1]['magpsf_ml'].values
            r_band_mag_corr = oid_detections[oid_detections.fid == 2]['magpsf_ml'].values

            g_band_mag = oid_detections[oid_detections.fid == 1]['magpsf'].values
            r_band_mag = oid_detections[oid_detections.fid == 2]['magpsf'].values

            g_r_max = g_band_mag.min() - r_band_mag.min()
            g_r_mean = g_band_mag.mean() - r_band_mag.mean()

            g_r_max_corr = g_band_mag_corr.min() - r_band_mag_corr.min()
            g_r_mean_corr = g_band_mag_corr.mean() - r_band_mag_corr.mean()

            if g_r_max == g_r_max_corr and g_r_mean == g_r_mean_corr:
                data = [g_r_max, g_r_mean, np.nan, np.nan]
            else:
                data = [g_r_max, g_r_mean, g_r_max_corr, g_r_mean_corr]
            oid_color = pd.Series(data=data, index=self.get_features_keys())
            return oid_color
        
        colors = detections.apply(aux_function)
        colors.index.name = 'oid'
        return colors
