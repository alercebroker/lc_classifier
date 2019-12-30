from late_classifier.features.extractors.sn_non_detections_features import SupernovaeNonDetectionFeatureComputer
from late_classifier.features.extractors.sn_detections_features import SupernovaeDetectionFeatureComputer
from late_classifier.features.extractors.galactic_coordinates import GalacticCoordinatesComputer
from late_classifier.features.extractors.turbofats_extractor import TurboFatsFeatureExtractor
from late_classifier.features.extractors.color_feature_computer import ColorFeatureComputer
from late_classifier.features.extractors.sg_score_computer import SGScoreComputer
from late_classifier.features.extractors.real_bogus_computer import RealBogusComputer
from late_classifier.features.extractors.paps_extractor import PAPSExtractor
from late_classifier.features.core.base import FeatureExtractor
from functools import reduce
import pandas as pd


class HierarchicalFeaturesComputer(FeatureExtractor):
    def __init__(self, bands):
        super().__init__()
        self.bands = bands
        self.turbofats_extractor = TurboFatsFeatureExtractor()
        self.sn_det_extractor = SupernovaeDetectionFeatureComputer()
        self.sn_nondet_extractor = SupernovaeNonDetectionFeatureComputer()
        self.galactic_coord_extractor = GalacticCoordinatesComputer()
        self.sgscore_extractor = SGScoreComputer()
        self.color_extractor = ColorFeatureComputer()
        self.rb_extractor = RealBogusComputer()
        self.paps_extractor = PAPSExtractor()
        self.features_keys = self.get_features_keys()

    def get_features_keys(self):
        return self.turbofats_extractor.features_keys + \
               self.sn_det_extractor.features_keys + \
               self.sn_nondet_extractor.features_keys + \
               self.galactic_coord_extractor.features_keys + \
               self.sgscore_extractor.features_keys + \
               self.color_extractor.features_keys + \
               self.rb_extractor.features_keys + \
               self.paps_extractor.feature_keys

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
        features = pd.DataFrame()

        for oid in oids:
            try:
                object_alerts = detections.loc[[oid]]
                if self.enough_alerts(object_alerts):
                    try:
                        object_non_detections = non_detections.loc[[oid]]
                    except KeyError:
                        object_non_detections = pd.DataFrame(columns=["fid", "mjd"])

                    for band in self.bands:
                        n_samples = len(object_alerts[object_alerts.fid == band])
                        enough_alerts_band = n_samples > 5
                        data_detections = object_alerts[object_alerts.fid == band]
                        data_non_detections = object_non_detections[object_non_detections.fid == band]
                        if enough_alerts_band:
                            sn_det_features = self.sn_nondet_extractor.compute_features(
                                data_detections, non_detections=data_non_detections)
                            turbofats_features = self.turbofats_extractor.compute_features(data_detections)
                            paps = self.paps_extractor.compute_features(detections)

                            df = sn_det_features.join(turbofats_features)
                            df = df.join(paps)

                            features = pd.concat([features, df], sort=True)
                        else:
                            df = pd.DataFrame([[band, oid]], columns=['fid', 'oid'])
                            df = df.set_index('oid')
                            features = features.append(df, sort=False)

            except Exception:
                raise Exception('Fatal error 0x187AF')
        return features

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
