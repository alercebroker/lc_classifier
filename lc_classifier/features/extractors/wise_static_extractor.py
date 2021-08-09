import os
from typing import Tuple
from functools import lru_cache

import requests
import pandas as pd
import logging
from ..core.base import FeatureExtractor


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
WISE_CSV = os.path.abspath(os.path.join(FILE_PATH, "data/wise_bands.csv"))
WISE_CSV_URL = "https://alerce-static.s3.amazonaws.com/datasets/wise_bands.csv"


def compute_colors_from_bands(bands: pd.DataFrame) -> pd.DataFrame:
    colors = bands.copy()
    colors["W1-W2"] = colors["W1"] - colors["W2"]
    colors["W2-W3"] = colors["W2"] - colors["W3"]
    colors["g-W2"] = colors["g"] - colors["W2"]
    colors["g-W3"] = colors["g"] - colors["W3"]
    colors["r-W2"] = colors["r"] - colors["W2"]
    colors["r-W3"] = colors["r"] - colors["W3"]
    colors = colors[["W1-W2", "W2-W3", "g-W2", "g-W3", "r-W2", "r-W3"]].copy()
    return colors


class WiseStaticExtractor(FeatureExtractor):
    def __init__(self):
        if not os.path.exists(WISE_CSV):
            data_dir = os.path.abspath(os.path.join(FILE_PATH, "data"))
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            wise_request = requests.get(WISE_CSV_URL)
            wise_request.raise_for_status()
            with open(WISE_CSV, "wb") as f:
                f.write(wise_request.content)
        self.wise_bands = pd.read_csv(WISE_CSV, index_col="oid")
        self.wise_colors = compute_colors_from_bands(self.wise_bands)

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return "W1-W2", "W2-W3", "g-W2", "g-W3", "r-W2", "r-W3"

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return ()

    def _compute_features(self, detections, **kwargs):
        oids = detections.index.unique()
        colors = []
        for oid in oids:
            if oid in self.wise_colors.index:
                colors.append(self.wise_colors.loc[[oid]])
            else:
                logging.debug(f"extractor=WISE object={oid} no xmatch")
                nan_df = self.nan_df(oid)
                nan_df.columns = self.get_features_keys()
                colors.append(nan_df)
        colors = pd.concat(colors, axis=0, sort=True)
        colors.index.name = "oid"
        return colors
