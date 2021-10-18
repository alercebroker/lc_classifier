from typing import Tuple
from functools import lru_cache

from ..core.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging


class ZTFColorFeatureExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'g-r_max', 'g-r_mean', 'g-r_max_corr', 'g-r_mean_corr'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'magpsf', 'magnitude'

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
            bands = oid_detections['band'].values
            unique_fids = np.unique(bands)
            if 1 not in unique_fids or 2 not in unique_fids:
                logging.debug(
                    f'extractor=COLOR  object={oid}  required_cols={self.get_required_keys()}  filters_qty=2')
                return self.nan_series()

            mag_corr = oid_detections['magnitude'].values
            g_band_mag_corr = mag_corr[bands == 1]
            r_band_mag_corr = mag_corr[bands == 2]

            mag = oid_detections['magpsf'].values
            g_band_mag = mag[bands == 1]
            r_band_mag = mag[bands == 2]

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


class ZTFColorForcedFeatureExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'g-r_max', 'g-r_mean'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'magnitude'

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

        def aux_function(oid_detections):
            oid = oid_detections.index.values[0]
            bands = oid_detections['band'].values
            unique_fids = np.unique(bands)
            if 1 not in unique_fids or 2 not in unique_fids:
                logging.debug(
                    f'extractor=COLOR  object={oid}  required_cols={self.get_required_keys()}  filters_qty=2')
                return self.nan_series()

            mag = oid_detections['magnitude'].values
            g_band_mag = mag[bands == 1]
            r_band_mag = mag[bands == 2]

            g_r_max = g_band_mag.min() - r_band_mag.min()
            g_r_mean = g_band_mag.mean() - r_band_mag.mean()

            data = [g_r_max, g_r_mean]
            oid_color = pd.Series(data=data, index=self.get_features_keys())
            return oid_color

        colors = detections.apply(aux_function)
        colors.index.name = 'oid'
        return colors
