from late_classifier.features.core.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging


class RealBogusExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.features_keys = ['rb']
        self.required_keys = ["rb"]

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
            logging.error(f'RB: Input dataframe invalid\n - Required columns: {self.required_keys}\n')
            return self.nan_df(index)

        rb = detections.rb.median()
        return pd.DataFrame(np.array([rb]), columns=self.features_keys, index=[index])
