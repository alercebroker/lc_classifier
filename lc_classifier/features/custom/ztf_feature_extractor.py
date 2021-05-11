from typing import Tuple

from lc_classifier.features import SupernovaeDetectionAndNonDetectionFeatureExtractor
from lc_classifier.features import GalacticCoordinatesExtractor
from lc_classifier.features import TurboFatsFeatureExtractor
from lc_classifier.features import ZTFColorFeatureExtractor
from lc_classifier.features import SGScoreExtractor, StreamSGScoreExtractor
from lc_classifier.features import RealBogusExtractor
from lc_classifier.features import MHPSExtractor
from lc_classifier.features import IQRExtractor
from lc_classifier.features import SNParametricModelExtractor
from lc_classifier.features import WiseStaticExtractor
from lc_classifier.features import WiseStreamExtractor
from lc_classifier.features import PeriodExtractor
from lc_classifier.features import PowerRateExtractor
from lc_classifier.features import FoldedKimExtractor
from lc_classifier.features import HarmonicsExtractor
from lc_classifier.features import GPDRWExtractor
from lc_classifier.features import SNFeaturesPhaseIIExtractor
from lc_classifier.features import SPMExtractorPhaseII


from ..core.base import FeatureExtractor
from ..core.base import FeatureExtractorComposer

import pandas as pd
from functools import lru_cache


class ZTFFeatureExtractor(FeatureExtractor):
    def __init__(self, bands=(1, 2), stream=False):
        self.bands = list(bands)
        self.stream = stream

        extractors = [
            GalacticCoordinatesExtractor(),
            ZTFColorFeatureExtractor(),
            RealBogusExtractor(),
            MHPSExtractor(bands),
            IQRExtractor(bands),
            TurboFatsFeatureExtractor(bands),
            SupernovaeDetectionAndNonDetectionFeatureExtractor(bands),
            SNParametricModelExtractor(bands),
            PeriodExtractor(bands=bands),
            PowerRateExtractor(bands),
            FoldedKimExtractor(bands),
            HarmonicsExtractor(bands),
            GPDRWExtractor(bands)
        ]
        if self.stream:
            extractors += [
                StreamSGScoreExtractor(),
                WiseStreamExtractor()
            ]
        else:
            extractors += [
                SGScoreExtractor(),
                WiseStaticExtractor()
            ]
        self.composed_feature_extractor = FeatureExtractorComposer(extractors)

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return self.composed_feature_extractor.get_features_keys()

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return self.composed_feature_extractor.get_required_keys()

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
        n_detections = detections[["time"]].groupby(level=0).count()
        has_enough_alerts = n_detections['time'] > 5
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
        required = ["non_detections"]
        if self.stream:
            required += ['metadata', 'xmatches']
        for key in required:
            if key not in kwargs:
                raise Exception(f"HierarchicalFeaturesComputer requires {key} argument")

        detections, too_short_oids = self.filter_out_short_lightcurves(detections)
        detections = detections.sort_values("time")

        non_detections = kwargs["non_detections"]
        if len(non_detections) == 0:
            non_detections = pd.DataFrame(columns=["time", "band", "diffmaglim"])

        shared_data = dict()

        kwargs['non_detections'] = non_detections
        kwargs['shared_data'] = shared_data

        features = self.composed_feature_extractor.compute_features(
            detections, **kwargs)

        too_short_features = pd.DataFrame(index=too_short_oids)
        df = pd.concat(
            [features, too_short_features],
            axis=0, join="outer", sort=True)
        return df

    def filter_out_short_lightcurves(self, detections):
        has_enough_alerts = self.get_enough_alerts_mask(detections)
        too_short_oids = has_enough_alerts[~has_enough_alerts]
        detections = detections.loc[has_enough_alerts]
        return detections, too_short_oids.index.values


class ZTFForcedPhotometryFeatureExtractor(FeatureExtractor):
    def __init__(self, bands=(1, 2), stream=False):
        self.bands = list(bands)
        self.stream = stream

        extractors = [
            GalacticCoordinatesExtractor(),
            ZTFColorFeatureExtractor(),
            RealBogusExtractor(),
            MHPSExtractor(bands),
            IQRExtractor(bands),
            TurboFatsFeatureExtractor(bands),
            SNFeaturesPhaseIIExtractor(bands),
            SPMExtractorPhaseII(bands),
            PeriodExtractor(bands=bands),
            PowerRateExtractor(bands),
            FoldedKimExtractor(bands),
            HarmonicsExtractor(bands),
            GPDRWExtractor(bands)
        ]
        if self.stream:
            extractors += [
                StreamSGScoreExtractor(),
                WiseStreamExtractor()
            ]
        else:
            extractors += [
                SGScoreExtractor(),
                WiseStaticExtractor()
            ]
        self.composed_feature_extractor = FeatureExtractorComposer(extractors)

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return self.composed_feature_extractor.get_features_keys()

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return self.composed_feature_extractor.get_required_keys()

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
        n_detections = detections[["time"]].groupby(level=0).count()
        has_enough_alerts = n_detections['time'] > 5
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
        required = []
        if self.stream:
            required += ['metadata', 'xmatches']
        for key in required:
            if key not in kwargs:
                raise Exception(f"HierarchicalFeaturesComputer requires {key} argument")

        detections, too_short_oids = self.filter_out_short_lightcurves(detections)
        detections = detections.sort_values("time")

        shared_data = dict()
        kwargs['shared_data'] = shared_data

        features = self.composed_feature_extractor.compute_features(
            detections, **kwargs)

        too_short_features = pd.DataFrame(index=too_short_oids)
        df = pd.concat(
            [features, too_short_features],
            axis=0, join="outer", sort=True)
        return df

    def filter_out_short_lightcurves(self, detections):
        has_enough_alerts = self.get_enough_alerts_mask(detections)
        too_short_oids = has_enough_alerts[~has_enough_alerts]
        detections = detections.loc[has_enough_alerts]
        return detections, too_short_oids.index.values