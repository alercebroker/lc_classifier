from typing import Tuple

from lc_classifier.features import TurboFatsFeatureExtractor
from lc_classifier.features import ElasticcColorFeatureExtractor
from lc_classifier.features import MHPSFluxExtractor
from lc_classifier.features import IQRExtractor
from lc_classifier.features import PeriodExtractor
from lc_classifier.features import PowerRateExtractor
from lc_classifier.features import FoldedKimExtractor
from lc_classifier.features import HarmonicsExtractor
from lc_classifier.features import GPDRWExtractor
from lc_classifier.features import SNFeaturesPhaseIIExtractor
from lc_classifier.features.extractors.sn_parametric_model_computer import SPMExtractorElasticc
from lc_classifier.features import ElasticcFullMetadataExtractor

from ..core.base import FeatureExtractor
from ..core.base import FeatureExtractorComposer

import pandas as pd
from functools import lru_cache


class ElasticcFeatureExtractor(FeatureExtractor):
    def __init__(self, round=1):
        self.bands = ['u', 'g', 'r', 'i', 'z', 'Y']

        magnitude_extractors = [
            # input: apparent magnitude
            IQRExtractor(self.bands),
            TurboFatsFeatureExtractor(self.bands),
            PeriodExtractor(
                bands=self.bands,
                smallest_period=0.045,
                largest_period=50.0,
                optimal_grid=True,
                trim_lightcurve_to_n_days=500.0,
                min_length=15
            ),
            PowerRateExtractor(self.bands),
            FoldedKimExtractor(self.bands),
            HarmonicsExtractor(self.bands),
            GPDRWExtractor(self.bands),
        ]

        flux_extractors = [
            # input: difference fluxd
            ElasticcFullMetadataExtractor(),
            ElasticcColorFeatureExtractor(self.bands),
            MHPSFluxExtractor(self.bands),
            SNFeaturesPhaseIIExtractor(self.bands),
            SPMExtractorElasticc(self.bands)
        ]
        
        self.magnitude_feature_extractor = FeatureExtractorComposer(
            magnitude_extractors)

        self.flux_feature_extractor = FeatureExtractorComposer(
            flux_extractors
        )

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return (
            self.magnitude_feature_extractor.get_features_keys()
            + self.flux_feature_extractor.get_features_keys()
        )

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return (
            self.magnitude_feature_extractor.get_required_keys()
            + self.flux_feature_extractor.get_required_keys()
        )

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
        required = ['metadata']
        for key in required:
            if key not in kwargs:
                raise Exception(f"{key} argument is missing")

        shared_data = dict()
        kwargs['shared_data'] = shared_data

        input_snids = detections.index.unique()

        magnitude_features = self.compute_magnitude_features(
            detections, **kwargs)

        flux_features = self.compute_flux_features(
            detections, **kwargs)

        df = pd.concat(
            # [flux_features],
            [magnitude_features, flux_features],
            axis=1, join="outer", sort=True)
        
        output_snids = df.index
        assert len(input_snids) == len(output_snids.intersection(input_snids))
        return df

    def compute_magnitude_features(self, detections, **kwargs):
        interesting_cols = [
            'time',
            'band',
            'magnitude',
            'error'
        ]

        detections, too_short_oids = self.filter_out_short_lightcurves(
            detections[interesting_cols])
        detections = detections.sort_values("time")

        magnitude_features = self.magnitude_feature_extractor.compute_features(
            detections, **kwargs
        )

        too_short_oids = [oid for oid in too_short_oids if oid not in magnitude_features.index.values]
        too_short_features = pd.DataFrame(index=too_short_oids)
        df = pd.concat(
            [magnitude_features, too_short_features],
            axis=0, join="outer", sort=True)
        return df

    def compute_flux_features(self, detections, **kwargs):
        interesting_cols = [
            'time',
            'band',
            'difference_flux',
            'difference_flux_error'
        ]

        detections, too_short_oids = self.filter_out_short_lightcurves(
            detections[interesting_cols])
        detections = detections.sort_values("time")

        flux_features = self.flux_feature_extractor.compute_features(
            detections, **kwargs
        )
        too_short_oids = [oid for oid in too_short_oids if oid not in flux_features.index.values]
        too_short_features = pd.DataFrame(index=too_short_oids)
        df = pd.concat(
            [flux_features, too_short_features],
            axis=0, join="outer", sort=True)
        return df

    def filter_out_short_lightcurves(self, detections):
        valid_alerts = detections.notna().all(axis=1)
        detections = detections[valid_alerts.values]

        has_enough_alerts = self.get_enough_alerts_mask(detections)
        too_short_oids = has_enough_alerts[~has_enough_alerts]
        detections = detections.loc[has_enough_alerts]
        return detections, too_short_oids.index.values
