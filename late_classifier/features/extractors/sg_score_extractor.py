from late_classifier.features.core.base import FeatureExtractor


class SGScoreComputer(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.features_keys = ['sgscore1']

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
        sgscore = detections['sgscore1'].groupby(level=0).median()
        return sgscore.to_frame()
