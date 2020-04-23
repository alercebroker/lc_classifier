import os
from typing import List

import requests
import pandas as pd
import logging
from ..core.base import FeatureExtractor


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
WISE_CSV = os.path.abspath(
    os.path.join(FILE_PATH, "data/wise_bands.csv"))
WISE_CSV_URL = 'https://droppy.alerce.online/$/5tZUY'


def compute_colors_from_bands(bands: pd.DataFrame) -> pd.DataFrame:
    colors = bands.copy()
    colors['W1-W2'] = colors['W1'] - colors['W2']
    colors['W2-W3'] = colors['W2'] - colors['W3']
    colors['g-W2'] = colors['g'] - colors['W2']
    colors['g-W3'] = colors['g'] - colors['W3']
    colors['r-W2'] = colors['r'] - colors['W2']
    colors['r-W3'] = colors['r'] - colors['W3']
    colors = colors[[
            'W1-W2',
            'W2-W3',
            'g-W2',
            'g-W3',
            'r-W2',
            'r-W3']].copy()
    return colors


class WiseStaticExtractor(FeatureExtractor):
    def __init__(self):
        if not os.path.exists(WISE_CSV):
            data_dir = os.path.abspath(
                os.path.join(FILE_PATH, "data"))
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            wise_request = requests.get(WISE_CSV_URL)
            wise_request.raise_for_status()
            with open(WISE_CSV, 'wb') as f:
                f.write(wise_request.content)
        self.wise_bands = pd.read_csv(WISE_CSV, index_col='oid')
        self.wise_colors = compute_colors_from_bands(self.wise_bands)

    def get_features_keys(self) -> List[str]:
        return [
            'W1-W2',
            'W2-W3',
            'g-W2',
            'g-W3',
            'r-W2',
            'r-W3'
        ]

    def get_required_keys(self) -> List[str]:
        return []

    def compute_features(self, detections, **kwargs):
        index = detections.index[0]
        if index in self.wise_colors.index:
            return self.wise_colors.loc[[index]]
        else:
            logging.warning(
                f'extractor=WISE  object={index}  required_cols={self.get_required_keys()}')
            nan_df = self.nan_df(index)
            nan_df.columns = self.get_features_keys()
            return nan_df
