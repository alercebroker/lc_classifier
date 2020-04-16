from late_classifier.features.core.base import FeatureExtractorSingleBand
import pandas as pd
import numpy as np
import logging
import mhps


class MHPSExtractor(FeatureExtractorSingleBand):
    def __init__(self, t1=100, t2=10, dt=3.0, mag0=19.0, epsilon=1.0):
        super().__init__()
        self.features_keys = [
            'mhps_ratio',
            'mhps_low',
            'mhps_high',
            'mhps_non_zero',
            'mhps_PN_flag']
        self.required_keys = ["magpsf_corr", "sigmapsf_corr", "mjd"]
        self.t1 = t1
        self.t2 = t2
        self.dt = dt
        self.mag0 = mag0
        self.epsilon = epsilon

    def _compute_features(self, detections, band=None, **kwargs):
        index = detections.index.unique()[0]
        columns = self.get_features_keys(band)
        detections = detections[detections.fid == band]

        if not self.validate_df(detections) or band is None or len(detections) == 0:
            logging.warning(f'extractor=MHPS  object={index}  required_cols={self.required_keys}  filters_qty=1')
            nan_df = self.nan_df(index)
            nan_df.columns = columns
            return nan_df
        mag = detections.magpsf_corr
        magerr = detections.sigmapsf_corr
        time = detections.mjd
        ratio, low, high, non_zero, PN_flag = mhps.statistics(mag.values,
                                                              magerr.values,
                                                              time.values,
                                                              self.t1,
                                                              self.t2)
        values = np.array([[ratio, low, high, non_zero, PN_flag]])
        df = pd.DataFrame(values, columns=columns, index=[index])
        return df
