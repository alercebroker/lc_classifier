from typing import List
import pandas as pd
from ..core.base import FeatureExtractor
from P4J import MultiBandPeriodogram
import logging


class PeriodExtractor(FeatureExtractor):
    def __init__(self):
        self.periodogram_computer = MultiBandPeriodogram(method='MHAOV')

    def get_features_keys(self) -> List[str]:
        return ['Multiband_period']

    def get_required_keys(self) -> List[str]:
        return [
            'mjd',
            'magpsf',
            'magpsf_corr',
            'fid',
            'sigmapsf_corr_ext',
            'sigmapsf']

    def _compute_features(self, detections: pd.DataFrame, **kwargs) -> pd.DataFrame:
        required = ['objects']
        for key in required:
            if key not in kwargs:
                raise Exception(f'Period extractor requires {key} argument')

        objects = kwargs['objects']

        oids = detections.index.unique()
        features = []

        detections = detections.sort_values('mjd')

        for oid in oids:
            oid_detections = detections.loc[[oid]]
            oid_objects = objects.loc[[oid]]

            # TODO: short light curves

            objects_corrected = oid_objects.corrected.values[0]
            if objects_corrected:
                self.feature_space.data_column_names = ['magpsf_corr', 'mjd', 'sigmapsf_corr_ext']
            else:
                self.feature_space.data_column_names = ['magpsf', 'mjd', 'sigmapsf']

            object_features = self.feature_space.calculate_features(oid_band_detections)
            object_features = pd.DataFrame(
                data=object_features.values,
                columns=[f'{c}_{band}' for c in object_features.columns],
                index=object_features.index
            )
            features.append(object_features)
        features = pd.concat(features, axis=0, sort=True)
        features.index.name = 'oid'
        return features