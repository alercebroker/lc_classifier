from typing import List

from late_classifier.features.core.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging


class SGScoreExtractor(FeatureExtractor):
    def get_features_keys(self) -> List[str]:
        return ['sgscore1']

    def get_required_keys(self) -> List[str]:
        return ['sgscore1']

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
            logging.warning(f'extractor=SGSCORE  object={index}  required_cols={self.get_required_keys()}')
            return self.nan_df(index)
        sgscore = detections['sgscore1'].median()
        return pd.DataFrame(np.array([sgscore]), columns=self.get_features_keys(), index=[index])
