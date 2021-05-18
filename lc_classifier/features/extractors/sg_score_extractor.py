from typing import Tuple
from functools import lru_cache

from ..core.base import FeatureExtractor
from pandas import DataFrame


class SGScoreExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'sgscore1',

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'sgscore1',

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections
        DataFrame with detections of an object.


        kwargs Not required.

        Returns :class:pandas.`DataFrame`
        -------

        """
        sgscore_medians = detections[['sgscore1']].median()
        return sgscore_medians


class StreamSGScoreExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'sgscore1',

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return ()

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
        metadata = kwargs["metadata"]
        oids = detections.index.unique()
        df_sgscore = metadata[["oid", "sgscore1"]]
        df_sgscore = df_sgscore.drop_duplicates("oid").set_index('oid')
        return df_sgscore.loc[oids]
