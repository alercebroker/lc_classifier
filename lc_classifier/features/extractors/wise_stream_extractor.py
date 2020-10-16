from typing import List, Tuple
import pandas as pd
from ..core.base import FeatureExtractor


class WiseStreamExtractor(FeatureExtractor):
    def __init__(self):
        super()
        self.xmatch_keys = ["W1mag", "W2mag", "W3mag"]

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

    def check_keys_xmatch(self, xmatch):
        if not xmatch:
            return True
        missing = set(xmatch.keys()).difference(self.get_required_keys())
        return len(missing) > 0

    def colors_set_null(self, colors, oid):
        colors["oid"] = [oid]
        colors["W1-W2"] = [None]
        colors["W2-W3"] = [None]
        colors["g-W2"] = [None]
        colors["g-W3"] = [None]
        colors["r-W2"] = [None]
        colors["r-W3"] = [None]

    def colors_set_values(self, colors, xmatch, g, r, oid):
        colors["oid"] = [oid]
        colors["W1-W2"] = [xmatch["W1mag"] - xmatch["W2mag"]]
        colors["W2-W3"] = [xmatch["W2mag"] - xmatch["W3mag"]]
        colors["g-W2"] = [g - xmatch["W2mag"]]
        colors["g-W3"] = [g - xmatch["W3mag"]]
        colors["r-W2"] = [r - xmatch["W2mag"]]
        colors["r-W3"] = [r - xmatch["W3mag"]]

    def _compute_features(self, detections, **kwargs):
        xmatch = kwargs["xmatches"]
        oid = detections.index.values[0]
        g, r = self.calculate_bands(detections)
        columns = self.get_features_keys()
        columns.append("oid")
        colors = pd.DataFrame(columns=columns)
        missing_xmatch = self.check_keys_xmatch(xmatch)
        if missing_xmatch:
            self.colors_set_null(colors, oid)
        else:
            self.colors_set_values(colors, xmatch, g, r, oid)
        colors.set_index("oid", inplace=True)
        return colors
