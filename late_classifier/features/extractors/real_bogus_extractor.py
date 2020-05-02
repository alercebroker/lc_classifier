from typing import List
from late_classifier.features.core.base import FeatureExtractor


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
        rb_medians = detections[['rb']].groupby(level=0).median()
        return rb_medians
