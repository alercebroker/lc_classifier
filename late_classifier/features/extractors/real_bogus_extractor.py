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

    def compute_features(self, detections, **kwargs):
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
        if not self.validate_df(detections):
            logging.warning(f'extractor=RB  object={index}  required_cols={self.get_required_keys()}')
            return self.nan_df(index)

        rb = detections.rb.median()
        return pd.DataFrame(np.array([rb]), columns=self.get_features_keys(), index=[index])
