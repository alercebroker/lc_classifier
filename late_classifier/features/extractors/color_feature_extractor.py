from typing import List

from ..core.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging


class ColorFeatureExtractor(FeatureExtractor):
    def get_features_keys(self) -> List[str]:
        return ['g-r_max', 'g-r_mean', 'g-r_max_corr', 'g-r_mean_corr']

    def get_required_keys(self) -> List[str]:
        return ['fid', 'magpsf_corr', 'magpsf']

    def _compute_features(self, detections, **kwargs):
        """
        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with detections of an object.

        objects : class:pandas.`DataFrame`
        Dataframe with the objects table.

        Returns :class:pandas.`DataFrame`
        -------
        """

        required = ['objects']
        for key in required:
            if key not in kwargs:
                raise Exception(f'MHPSExtractor requires {key} argument')

        objects = kwargs['objects']

        # pd.options.display.precision = 10
        oids = detections.index.unique()
        colors = []
        for oid in oids:
            oid_detections = detections.loc[[oid]]
            oid_objects = objects.loc[[oid]]

            if 1 not in oid_detections.fid.unique() or 2 not in oid_detections.fid.unique():
                logging.info(
                    f'extractor=COLOR  object={oid}  '
                    f'required_cols={self.get_required_keys()}  filters_qty=2')
                colors.append(self.nan_df(oid))
                continue

            objects_corrected = oid_objects.corrected.values[0]

            g_band_mag_corr = oid_detections[oid_detections.fid == 1]['magpsf_corr'].values
            r_band_mag_corr = oid_detections[oid_detections.fid == 2]['magpsf_corr'].values

            g_band_mag = oid_detections[oid_detections.fid == 1]['magpsf'].values
            r_band_mag = oid_detections[oid_detections.fid == 2]['magpsf'].values

            g_r_max = g_band_mag.min() - r_band_mag.min()
            g_r_mean = g_band_mag.mean() - r_band_mag.mean()

            if objects_corrected:
                g_r_max_corr = g_band_mag_corr.min() - r_band_mag_corr.min()
                g_r_mean_corr = g_band_mag_corr.mean() - r_band_mag_corr.mean()

            else:
                g_r_max_corr = np.nan
                g_r_mean_corr = np.nan

            data = [g_r_max, g_r_mean, g_r_max_corr, g_r_mean_corr]
            oid_color = pd.DataFrame([data], columns=self.get_features_keys(), index=[oid])
            colors.append(oid_color)
        colors = pd.concat(colors, axis=0)
        colors.index.name = 'oid'
        return colors
