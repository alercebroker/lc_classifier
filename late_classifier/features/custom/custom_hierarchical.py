from typing import List

from late_classifier.features import SupernovaeDetectionAndNonDetectionFeatureExtractor
from late_classifier.features import GalacticCoordinatesExtractor
from late_classifier.features import TurboFatsFeatureExtractor
from late_classifier.features import ColorFeatureExtractor
from late_classifier.features import SGScoreExtractor
from late_classifier.features import RealBogusExtractor
from late_classifier.features import MHPSExtractor
from late_classifier.features import IQRExtractor
from late_classifier.features import SNParametricModelExtractor
from late_classifier.features import WiseStaticExtractor
from late_classifier.features import PeriodExtractor
from late_classifier.features import PowerRateExtractor
from late_classifier.features import FoldedKimExtractor
from late_classifier.features import HarmonicsExtractor

from ..core.base import FeatureExtractor, FeatureExtractorSingleBand
from ..preprocess import DetectionsPreprocessorZTF

import pandas as pd
import logging


class CustomHierarchicalExtractor(FeatureExtractor):
    def __init__(self, bands=None):
        self.bands = bands if bands is not None else [1, 2]
        self.extractors = [GalacticCoordinatesExtractor(),
                           SGScoreExtractor(),
                           ColorFeatureExtractor(),
                           RealBogusExtractor(),
                           MHPSExtractor(),
                           IQRExtractor(),
                           TurboFatsFeatureExtractor(),
                           SupernovaeDetectionAndNonDetectionFeatureExtractor(),
                           SNParametricModelExtractor(),
                           WiseStaticExtractor(),
                           PeriodExtractor(),
                           PowerRateExtractor(),
                           FoldedKimExtractor(),
                           HarmonicsExtractor()
                           ]
        self.preprocessor = DetectionsPreprocessorZTF()

    def get_features_keys(self) -> List[str]:
        features_keys = []
        for extractor in self.extractors:
            if isinstance(extractor, FeatureExtractorSingleBand):
                for band in [1, 2]:
                    features_keys.append(extractor.get_features_keys_with_band(band))
            else:
                features_keys.append(extractor.get_features_keys())
        return features_keys

    def get_required_keys(self) -> List[str]:
        required_keys = set()
        for extractor in self.extractors:
            required_keys.union(set(extractor.get_required_keys()))
        return list(required_keys)

    def get_enough_alerts_mask(self, detections):
        """
        Verify if an object has enough alerts.
        Parameters
        ----------
        object_alerts: :class:pandas. `DataFrame`
        DataFrame with detections of only one object.

        Returns Boolean
        -------

        """
        n_detections = detections[['mjd']].groupby(level=0).count()
        has_enough_alerts = n_detections.mjd > 5
        return has_enough_alerts

    def _compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        kwargs Possible non_detections :class:pandas.`DataFrame`
                        objects :class:pandas.`DataFrame`

        Returns DataFrame with all features
        -------

        """
        required = ['non_detections', 'objects']
        for key in required:
            if key not in kwargs:
                raise Exception(f'HierarchicalFeaturesComputer requires {key} argument')
        objects = kwargs['objects']
        detections = self.preprocessor.preprocess(detections, objects=objects)
        has_enough_alerts = self.get_enough_alerts_mask(detections)
        too_short_oids = has_enough_alerts[~has_enough_alerts]
        too_short_features = pd.DataFrame(index=too_short_oids.index)
        detections = detections.loc[has_enough_alerts]
        detections = detections.sort_values('mjd')
        non_detections = kwargs['non_detections']

        if len(non_detections) == 0:
            non_detections = pd.DataFrame(columns=["mjd", "fid", "diffmaglim"])

        features = []
        shared_data = dict()
        for ex in self.extractors:
            df = ex.compute_features(
                detections,
                non_detections=non_detections,
                shared_data=shared_data
            )
            logging.info(f'FLAG={ex}')
            features.append(df)
        df = pd.concat(features, axis=1, join='inner')
        df = pd.concat([df, too_short_features], axis=0, join='outer', sort=True)
        return df
