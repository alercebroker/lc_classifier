from late_classifier.features.core.base import FeatureExtractor


class RealBogusComputer(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.features_keys = ['rb']

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
        rb = detections['rb'].groupby(level=0).median()
        rb.rename('rb', inplace=True)
        return rb.to_frame()
