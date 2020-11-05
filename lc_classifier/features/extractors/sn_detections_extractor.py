from typing import List

from ..core.base import FeatureExtractorSingleBand
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
        return ["isdiffpos", "magpsf_ml", "mjd"]

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
            if band not in oid_detections.fid.values:
                logging.info(
                    f'extractor=SN detection object={oid} required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)

            oid_band_detections = oid_detections[oid_detections.fid == band].sort_values('mjd')

            n_pos = len(oid_band_detections[oid_band_detections.isdiffpos > 0])
            n_neg = len(oid_band_detections[oid_band_detections.isdiffpos < 0])
            min_mag = oid_band_detections['magpsf_ml'].values.min()
            first_mag = oid_band_detections['magpsf_ml'].values[0]
            delta_mjd_fid = oid_band_detections['mjd'].values[-1] - \
                oid_band_detections['mjd'].values[0]
            delta_mag_fid = oid_band_detections['magpsf_ml'].values.max(
            ) - min_mag
            positive_fraction = n_pos/(n_pos + n_neg)
            mean_mag = oid_band_detections['magpsf_ml'].values.mean()

            data = [delta_mag_fid,
                    delta_mjd_fid,
                    first_mag,
                    mean_mag,
                    min_mag,
                    n_neg + n_pos,
                    n_neg,
                    n_pos,
                    positive_fraction]
            sn_det_df = pd.Series(
                data=data,
                index=columns)
            return sn_det_df

        sn_det_results = detections.apply(aux_function)
        sn_det_results.index.name = 'oid'
        return sn_det_results
