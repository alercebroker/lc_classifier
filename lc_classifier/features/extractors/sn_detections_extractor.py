from typing import Tuple
from functools import lru_cache

from ..core.base import FeatureExtractorSingleBand
import pandas as pd
import logging


class SupernovaeDetectionFeatureExtractor(FeatureExtractorSingleBand):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return ('delta_mag_fid',
                'delta_mjd_fid',
                'first_mag',
                'mean_mag',
                'min_mag',
                'n_det',
                'n_neg',
                'n_pos',
                'positive_fraction')

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return "isdiffpos", "magpsf_ml", "mjd", "fid"

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs): 
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
        
        columns = self.get_features_keys_with_band(band)

        def aux_function(oid_detections, **kwargs):
            fids = oid_detections['fid'].values
            if band not in fids:
                oid = oid_detections.index.values[0]
                logging.debug(
                    f'extractor=SN detection object={oid} required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)

            oid_band_detections = oid_detections[fids == band].sort_values('mjd')

            is_diff_pos_mask = oid_band_detections['isdiffpos'] > 0
            n_pos = len(oid_band_detections[is_diff_pos_mask])
            n_neg = len(oid_band_detections[~is_diff_pos_mask])

            mags = oid_band_detections['magpsf_ml'].values
            min_mag = mags.min()
            first_mag = mags[0]

            mjds = oid_band_detections['mjd'].values
            delta_mjd_fid = mjds[-1] - mjds[0]
            delta_mag_fid = mags.max() - min_mag
            positive_fraction = n_pos/(n_pos + n_neg)
            mean_mag = mags.mean()

            data = [
                delta_mag_fid,
                delta_mjd_fid,
                first_mag,
                mean_mag,
                min_mag,
                n_neg + n_pos,
                n_neg,
                n_pos,
                positive_fraction
            ]

            sn_det_df = pd.Series(
                data=data,
                index=columns)
            return sn_det_df

        sn_det_results = detections.apply(aux_function)
        sn_det_results.index.name = 'oid'
        return sn_det_results
