from typing import List

from ..core.base import FeatureExtractor
from pandas import DataFrame


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


class StreamSGScoreExtractor(FeatureExtractor):
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
        sgscore = kwargs["metadata"]["ps1"]["sgscore1"]
        df_sgscore = DataFrame({"sgscore1": [sgscore]})
        return df_sgscore
