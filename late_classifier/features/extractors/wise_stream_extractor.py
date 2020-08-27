from typing import List, Tuple
import pandas as pd
from ..core.base import FeatureExtractor


class WiseStreamExtractor(FeatureExtractor):
    def get_features_keys(self) -> List[str]:
        return ["W1-W2", "W2-W3", "g-W2", "g-W3", "r-W2", "r-W3"]

    def get_required_keys(self) -> List[str]:
        return []

    def calculate_bands(self, detections) -> Tuple:
        """
        Calculates the g and r bands.

        Parameters
        ----------
        detections : pd.DataFrame
            detections

        Returns g, r
        """
        g = detections[detections["fid"] == 1]["sigmapsf_corr"].mean(axis=0)
        r = detections[detections["fid"] == 2]["sigmapsf_corr"].mean(axis=0)
        return g, r

    def _compute_features(self, detections, **kwargs):
        xmatch = kwargs["xmatch"]
        g, r = self.calculate_bands(detections)
        colors = pd.DataFrame(columns=self.get_features_keys())
        colors["W1-W2"] = [xmatch["W1mag"] - xmatch["W2mag"]]
        colors["W2-W3"] = [xmatch["W2mag"] - xmatch["W3mag"]]
        colors["g-W2"] = [g - xmatch["W2mag"]]
        colors["g-W3"] = [g - xmatch["W3mag"]]
        colors["r-W2"] = [r - xmatch["W2mag"]]
        colors["r-W3"] = [r - xmatch["W3mag"]]
        return colors
