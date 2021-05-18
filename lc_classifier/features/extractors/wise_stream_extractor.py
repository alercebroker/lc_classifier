from typing import Tuple
from functools import lru_cache
import pandas as pd
import numpy as np
from ..core.base import FeatureExtractor
import logging


class WiseStreamExtractor(FeatureExtractor):
    def __init__(self):
        super()
        self.xmatch_keys = ["W1mag", "W2mag", "W3mag"]

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return "W1-W2", "W2-W3", "g-W2", "g-W3", "r-W2", "r-W3"

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'magnitude'

    def calculate_bands(self, detections) -> Tuple:
        """
        Calculates the g and r bands.

        Parameters
        ----------
        detections : pd.DataFrame
            detections

        Returns g, r
        """
        assert 'magnitude' in detections.columns
        
        detections_g = detections[detections["band"] == 1]
        detections_r = detections[detections["band"] == 2]
        g = detections_g["magnitude"].groupby(detections_g.index).mean()
        g.name = "g"
        r = detections_r["magnitude"].groupby(detections_r.index).mean()
        r.name = "r"
        return g, r

    def check_keys_xmatch(self, xmatch):
        missing = []
        for key in self.xmatch_keys:
            if key not in xmatch.columns:
                missing.append(key)
        if len(missing) > 0:
            logging.info(f"Missing keys: {missing}")
        return len(missing) > 0

    def colors_set_null(self, colors, oid):
        colors["oid"] = oid
        colors["W1-W2"] = np.nan
        colors["W2-W3"] = np.nan
        colors["g-W2"] = np.nan
        colors["g-W3"] = np.nan
        colors["r-W2"] = np.nan
        colors["r-W3"] = np.nan

    def colors_set_values(self, xmatch, g, r):
        g_r_wise = xmatch.join(g).join(r)
        g_r_wise["W1-W2"] = g_r_wise["W1mag"] - g_r_wise["W2mag"]
        g_r_wise["W2-W3"] = g_r_wise["W2mag"] - g_r_wise["W3mag"]
        g_r_wise["g-W2"] = g_r_wise["g"] - g_r_wise["W2mag"]
        g_r_wise["g-W3"] = g_r_wise["g"] - g_r_wise["W3mag"]
        g_r_wise["r-W2"] = g_r_wise["r"] - g_r_wise["W2mag"]
        g_r_wise["r-W3"] = g_r_wise["r"] - g_r_wise["W3mag"]
        return g_r_wise

    def _compute_features(self, detections, **kwargs):
        xmatch = kwargs["xmatches"]
        oids = detections.index.unique()
        g, r = self.calculate_bands(detections)
        columns = list(self.get_features_keys())
        columns.append("oid")
        missing_xmatch = self.check_keys_xmatch(xmatch)
        xmatch.set_index("oid", inplace=True)

        if missing_xmatch:
            logging.info("Missing xmatch features")
            colors = pd.DataFrame(columns=columns)
            self.colors_set_null(colors, oids)
            colors.set_index("oid", inplace=True)
        else:
            colors = self.colors_set_values(xmatch, g, r)
            colors = colors.loc[~colors.index.duplicated()]
            colors = colors.loc[oids, self.get_features_keys()]
        return colors
