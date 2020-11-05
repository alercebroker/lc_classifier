from typing import List
from ..core.base import FeatureExtractor


class RealBogusExtractor(FeatureExtractor):
    def get_features_keys(self) -> List[str]:
        return ['rb']

    def get_required_keys(self) -> List[str]:
        return ['rb']

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)
    
    def _compute_features_from_df_groupby(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections
        kwargs

        Returns class:pandas.`DataFrame`
        Real Bogus features.
        -------

        """
        return detections[['rb']].median()
