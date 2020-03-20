from late_classifier.features.core.base import FeatureExtractorSingleBand
import pandas as pd
import numpy as np
import mhps


class MHPSExtractor(FeatureExtractorSingleBand):
    def __init__(self, t1=100, t2=10, dt=3.0, mag0=19.0, epsilon=1.0):
        super().__init__()
        self.feature_keys = [
            'mhps_ratio',
            'mhps_low',
            'mhps_high',
            'mhps_non_zero',
            'mhps_PN_flag']
        
        self.t1 = t1
        self.t2 = t2
        self.dt = dt
        self.mag0 = mag0
        self.epsilon = epsilon

    def compute_features(self, detections, **kwargs):
        mag = detections.magpsf_corr
        magerr = detections.sigmapsf_corr
        time = detections.mjd
        ratio, low, high, non_zero, PN_flag = mhps.statistics(mag.values,
                                                              magerr.values,
                                                              time.values,
                                                              self.t1,
                                                              self.t2)
        oid = detections.index.unique()[0]
        values = np.array([[ratio, low, high, non_zero, PN_flag]])
        df = pd.DataFrame(values, columns=self.feature_keys, index=[oid])
        return df
