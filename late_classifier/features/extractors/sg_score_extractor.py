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

    def _compute_features(self, detections, **kwargs):
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

        sgscore = detections['sgscore1'].median()
        return pd.DataFrame(np.array([sgscore]), columns=self.get_features_keys(), index=[index])
