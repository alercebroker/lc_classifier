from late_classifier.features.core.base import FeatureExtractorSingleBand
import scipy.stats as sstats
import pandas as pd
import logging


class IQRExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        super().__init__()
        self.features_keys = ["iqr"]
        self.required_keys = ["magpsf_corr"]

    def _compute_features(self, detections, band=None, **kwargs):
        index = detections.index[0]
        columns = self.get_features_keys(band)

        if not self.validate_df(detections) or band is None or len(detections.fid.unique()) != 1:
            logging.error(
                f'Input dataframe invalid\n - Required columns: {self.required_keys}\n - Required one filter.')
            nan_df = self.nan_df(index)
            nan_df.columns = columns
            return nan_df
        detections = detections[detections.fid == band]
        detections = detections.sort_values('mjd')
        mag_dets = detections["magpsf_corr"]
        iqr = sstats.iqr(mag_dets.values)
        return pd.DataFrame([iqr], columns=columns)
