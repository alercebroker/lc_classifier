from typing import List

from late_classifier.features.core.base import FeatureExtractorSingleBand
import scipy.stats as sstats
import pandas as pd
import logging


class IQRExtractor(FeatureExtractorSingleBand):
    def get_features_keys(self) -> List[str]:
        return ['iqr']

    def get_required_keys(self) -> List[str]:
        return ['magpsf_corr']

    def compute_feature_in_one_band(self, detections, band=None, **kwargs):
        index = detections.index[0]
        columns = self.get_features_keys_with_band(band)

        if band is None:
            logging.warning(f'extractor=IQR  object={index}  required_cols={self.get_required_keys()}  filters_qty=1')
            nan_df = self.nan_df(index)
            nan_df.columns = columns
            return nan_df
        detections = detections[detections.fid == band]
        detections = detections.sort_values('mjd')
        mag_dets = detections["magpsf_corr"]
        iqr = sstats.iqr(mag_dets.values)
        return pd.DataFrame([iqr], columns=columns, index=[index])
