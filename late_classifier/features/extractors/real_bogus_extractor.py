from typing import List

from late_classifier.features.core.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging


class RealBogusExtractor(FeatureExtractor):
    def get_features_keys(self) -> List[str]:
        return ['rb']

    def get_required_keys(self) -> List[str]:
        return ['rb']

    def _compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        kwargs

        Returns class:pandas.`DataFrame`
        Real Bogus features.
        -------

        """
        index = detections.index[0]

        rb = detections.rb.median()
        return pd.DataFrame(np.array([rb]), columns=self.get_features_keys(), index=[index])
