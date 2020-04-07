from late_classifier.features.extractors.sn_non_detections_extractor import SupernovaeNonDetectionFeatureExtractor
from late_classifier.features.extractors.sn_detections_extractor import SupernovaeDetectionFeatureExtractor
from late_classifier.features.extractors.galactic_coordinates_extractor import GalacticCoordinatesExtractor
from late_classifier.features.extractors.turbofats_extractor import TurboFatsFeatureExtractor
from late_classifier.features.extractors.color_feature_extractor import ColorFeatureExtractor
from late_classifier.features.extractors.sg_score_extractor import SGScoreExtractor
from late_classifier.features.extractors.real_bogus_extractor import RealBogusExtractor
from late_classifier.features.extractors.mhps_extractor import MHPSExtractor
from late_classifier.features.extractors.iqr_extractor import IQRExtractor

from late_classifier.features.core.base import FeatureExtractor
from functools import reduce
import pandas as pd


class CustomHierarchicalExtractor(FeatureExtractor):
    def __init__(self, bands=None):
        super().__init__()
        self.bands = bands if bands is not None else [1, 2]
        self.extractors = [GalacticCoordinatesExtractor(),
                           SGScoreExtractor(),
                           ColorFeatureExtractor(),
                           RealBogusExtractor(),
                           MHPSExtractor(),
                           IQRExtractor(),
                           TurboFatsFeatureExtractor(),
                           #SupernovaeDetectionFeatureExtractor(),
                           SupernovaeNonDetectionFeatureExtractor()]

    def enough_alerts(self, object_alerts):
        """
        Verify if an object has enough alerts.
        Parameters
        ----------
        object_alerts: :class:pandas. `DataFrame`
        DataFrame with detections of only one object.

        Returns Boolean
        -------

        """
        if len(object_alerts) <= 0 and type(object_alerts) is not pd.DataFrame and len(object_alerts.index.unique()) == 1:
            return False
        booleans = list(map(lambda band: len(object_alerts.fid == band) > 5, self.bands))
        return reduce(lambda x, y: x | y, booleans)

    def compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        kwargs Possible non_detections :class:pandas.`DataFrame`

        Returns DataFrame with all features
        -------

        """
        required = ['non_detections']
        for key in required:
            if key not in kwargs:
                raise Exception(f'HierarchicalFeaturesComputer requires {key} argument')
        if type(detections) is not pd.DataFrame:
            raise Exception(
                f'HierarchicalFeaturesComputer requires detections in pd.Dataframe datatype, not {type(detections)}')

        detections = detections.sort_values('mjd')
        non_detections = kwargs['non_detections']

        if len(non_detections) == 0:
            non_detections = pd.DataFrame(columns=["mjd", "fid", "diffmaglim"])

        features = []
        for ex in self.extractors:
            df = ex.compute_features(detections, non_detections=non_detections)
            features.append(df)
        df = pd.concat(features, axis=1, join='inner')
        return df
