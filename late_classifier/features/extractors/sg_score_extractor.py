from late_classifier.features.core.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging


class SGScoreExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.features_keys = ['sgscore1']
        self.required_keys = ["sgscore1"]

    def compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with detections of an object.


        kwargs Not required.

        Returns :class:pandas.`DataFrame`
        -------

        """
        index = detections.index[0]
        if not self.validate_df(detections):
            logging.warning(f'extractor=SGSCORE  object={index}  required_cols={self.required_keys}')
            return self.nan_df(index)
        sgscore = detections['sgscore1'].median()
        return pd.DataFrame(np.array([sgscore]), columns=self.features_keys, index=[index])
