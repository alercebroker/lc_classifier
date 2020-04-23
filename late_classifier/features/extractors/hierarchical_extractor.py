from typing import List

from late_classifier.features.extractors.sn_non_detections_extractor import SupernovaeNonDetectionFeatureExtractor
from late_classifier.features.extractors.sn_detections_extractor import SupernovaeDetectionFeatureExtractor
from late_classifier.features.extractors.galactic_coordinates_extractor import GalacticCoordinatesExtractor
from late_classifier.features.extractors.turbofats_extractor import TurboFatsFeatureExtractor
from late_classifier.features.extractors.color_feature_extractor import ColorFeatureExtractor
from late_classifier.features.extractors.sg_score_extractor import SGScoreExtractor
from late_classifier.features.extractors.real_bogus_extractor import RealBogusExtractor
from late_classifier.features.extractors.mhps_extractor import MHPSExtractor
from late_classifier.features.extractors.iqr_extractor import IQRExtractor
from late_classifier.features.extractors.sn_parametric_model_computer import SNParametricModelExtractor

from late_classifier.features.core.base import FeatureExtractor
from functools import reduce
import pandas as pd
import warnings

# TODO deprecate this in favor of custom_hierarchical


class HierarchicalExtractor(FeatureExtractor):
    def __init__(self, bands):
        self.bands = bands
        self.turbofats_extractor = TurboFatsFeatureExtractor()
        self.sn_det_extractor = SupernovaeNonDetectionFeatureExtractor()
        self.sn_nondet_extractor = SupernovaeDetectionFeatureExtractor()
        self.galactic_coord_extractor = GalacticCoordinatesExtractor()
        self.sgscore_extractor = SGScoreExtractor()
        self.color_extractor = ColorFeatureExtractor()
        self.rb_extractor = RealBogusExtractor()
        self.paps_extractor = MHPSExtractor()
        self.iqr_extractor = IQRExtractor()
        self.sn_model_extractor = SNParametricModelExtractor()
        
        self.features_keys = self.get_features_keys()

    def get_required_keys(self) -> List[str]:
        return ['mjd', 'magpsf_corr', 'sigmapsf_corr', 'fid']

    def get_features_keys(self):
        return self.turbofats_extractor.get_features_keys() + \
               self.sn_det_extractor.get_features_keys() + \
               self.sn_nondet_extractor.get_features_keys() + \
               self.galactic_coord_extractor.get_features_keys() + \
               self.sgscore_extractor.get_features_keys() + \
               self.color_extractor.get_features_keys() + \
               self.rb_extractor.get_features_keys() + \
               self.paps_extractor.get_features_keys() + \
               self.iqr_extractor.get_features_keys()

    def enough_alerts(self, object_alerts: pd.DataFrame):
        """
        Verify if an object has enough alerts.
        Parameters
        ----------
        object_alerts: :class:pandas. `DataFrame`
        DataFrame with detections of only one object.
        Returns Boolean
        -------
        """
        if (
                len(object_alerts) <= 0
                and type(object_alerts) is not pd.DataFrame
                and len(object_alerts.index.unique()) == 1):
            return False
        booleans = list(map(lambda band: len(object_alerts.fid == band) > 5, self.bands))
        return reduce(lambda x, y: x | y, booleans)

    def single_band_features(self, detections, non_detections):
        """
        Compute a single band features of detections and non detections. Given a DataFrame of
        different objects this extract single band features for each objects.
        Parameters
        ----------
        detections: :class:pandas.`DataFrame`
        DataFrame that contains detections of different astronomical objects.
        non_detections: :class:pandas.`DataFrame`
        DataFrame that contains non detections of different an astronomical objects.
        Returns DataFrame with single band features joined
        -------
        """
        oids = detections.index.unique()
        features = []

        for oid in oids:
            try:
                object_alerts = detections.loc[[oid]]
                if self.enough_alerts(object_alerts):
                    try:
                        object_non_detections = non_detections.loc[[oid]]
                    except KeyError:
                        object_non_detections = pd.DataFrame(columns=["fid", "mjd"])

                    data_detections = object_alerts
                    data_non_detections = object_non_detections
                    sn_det_features = self.sn_nondet_extractor.compute_features(
                        data_detections, non_detections=data_non_detections)
                    turbofats_features = self.turbofats_extractor.compute_features(data_detections)
                    paps = self.paps_extractor.compute_features(data_detections)
                    iqr = self.iqr_extractor.compute_features(data_detections)

                    df = sn_det_features.join(turbofats_features)
                    df = df.join(paps)
                    df = df.join(iqr)
            except Exception as err:
                raise Exception(f'Fatal error at hierarchical_extractor: {err}')
        return pd.concat(features)

    def multi_band_features(self, detections):
        """
        Extraction's features of multi band detections:
            - Galactic Coordinates
            - SG Score
            - Color
            - Real Bogus
        Parameters
        ----------
        detections: :class:pandas.`DataFrame`
        DataFrame that contains detections of different astronomical objects.
        Returns DataFrame with multi band features joined
        -------
        """
        galactic = self.galactic_coord_extractor.compute_features(detections)
        sg_score = self.sgscore_extractor.compute_features(detections)
        color = self.color_extractor.compute_features(detections)
        rb = self.rb_extractor.compute_features(detections)

        df = galactic.join(sg_score)
        df = df.join(color)
        df = df.join(rb)
        return df

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
        non_detections = kwargs['non_detections'].sort_values('mjd')

        multi = self.multi_band_features(detections)
        single = self.single_band_features(detections, non_detections)

        print(single.columns)

        single_band_features = [f for f in single.columns.values if f != 'fid']

        if len(single) > 0:
            fids = single.fid.unique()
            single_band_dfs = []
            for fid in fids:
                features_band_fid = single[single.fid == fid]
                features_band_fid = features_band_fid[single_band_features]
                features_band_fid = features_band_fid.rename(
                    lambda x: x + f'_{fid}',
                    axis='columns')
                single_band_dfs.append(features_band_fid)
            single = pd.concat(single_band_dfs, axis=1, join='outer', sort=True)
            df = single.join(multi)
            return df
        else:
            return pd.DataFrame([])
