import scipy.stats as sstats
import pandas as pd
from late_classifier.features.core.base import FeatureExtractorSingleBand

class IQRExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        super().__init__()
        self.features_keys = ["iqr"]

    def _compute_features(self, detections, **kwargs):
        detections = detections.sort_values('mjd')
        if(len(detections) != 0):
            mag_dets = detections["magpsf_corr"]
            iqr = sstats.iqr(mag_dets.values)
            my_features = {
                'oid': detections.index[0],
                'iqr': iqr
            }
            return pd.DataFrame.from_records([my_features], index='oid')
        return pd.DataFrame()
