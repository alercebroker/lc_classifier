from typing import List

from ..core.base import FeatureExtractor


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
        sgscore_medians = detections[['sgscore1']].groupby(level=0).median()
        return sgscore_medians
