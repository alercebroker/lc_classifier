from typing import List

from ..core.base import FeatureExtractorSingleBand
import scipy.stats as sstats
import pandas as pd
import logging


class IQRExtractor(FeatureExtractorSingleBand):
    def get_features_keys(self) -> List[str]:
        return ['iqr']

    def get_required_keys(self) -> List[str]:
        return ['fid', 'magpsf_ml']

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band, **kwargs)

    def compute_feature_in_one_band_from_group(
            self, detections, band, **kwargs):

        columns = self.get_features_keys_with_band(band)

        def aux_function(oid_detections, **kwargs):
            if band not in oid_detections.fid.values:
                oid = oid_detections.index.values[0]
                logging.info(
                    f'extractor=IQR  object={oid}  required_cols={self.get_required_keys()}  band={band}')
                return self.nan_series_in_band(band)
            
            oid_band_detections = oid_detections[oid_detections.fid == band]
            mag_dets = oid_band_detections["magpsf_ml"]
            iqr = sstats.iqr(mag_dets.values)
            return pd.Series(
                data=[iqr],
                index=columns)
        
        iqrs = detections.apply(aux_function)
        iqrs.index.name = 'oid'
        return iqrs
