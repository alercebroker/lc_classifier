import pandas as pd
from typing import Tuple
from functools import lru_cache
from ..core.base import FeatureExtractor


class ElasticcMetadataExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'redshift_helio', 'mwebv'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        # not very useful
        return tuple()

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(
            self, detections, **kwargs) -> pd.DataFrame:
        columns = self.get_features_keys()
        metadata = kwargs['metadata']

        def aux_function(oid_detections, **kwargs):
            oid = oid_detections.index.values[0]
            metadata_lightcurve = metadata.loc[oid]
            redshift_helio = metadata_lightcurve['REDSHIFT_HELIO']
            mwebv = metadata_lightcurve['MWEBV']

            out = pd.Series(
                data=[redshift_helio, mwebv],
                index=columns)
            return out

        features = detections.apply(aux_function)
        features.index.name = 'oid'
        return features
