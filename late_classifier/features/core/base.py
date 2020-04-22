# from .decorators import *
import pandas as pd


class FeatureExtractor:
    def __init__(self):
        self.features_keys = []
        self.required_keys = []

    def validate_df(self, df):
        """
        Method to validate input detections. The required keys must be in df columns.

        Parameters
        ----------
        df :class:pandas.`DataFrame`
        DataFrame with detections of an object.
        """
        cols = set(df.columns)
        required = set(self.required_keys)
        intersection = required.intersection(cols)
        return len(intersection) == len(required)

    def nan_df(self, index):
        """
        Method to generate a empty dataframe with NaNs.

        Parameters
        ----------
        index :class:String
        Name/id/oid of the observation
        """
        return pd.DataFrame(columns=self.features_keys, index=[index])

    def compute_features(self, detections: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Interface to implement different features extractors.

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with detections of an object.

        kwargs Another arguments like Non detections.
        """
        raise NotImplementedError('FeatureExtractor is an interface')


class FeatureExtractorSingleBand(FeatureExtractor):
    def compute_features(self, detections, **kwargs):
        """
        Compute features to single band detections of an object. Verify if input has only one band.

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        Single band detections of an object.

        kwargs Possible required: Non detections.

        Returns :class:pandas.`DataFrame`
        -------

        """
        return self.compute_by_bands(detections, **kwargs)

    def _compute_features(self, detections, **kwargs):
        """
        Interface to compute features to single band detections of an object.
        Parameters
        ----------
        detections :class:pandas.`DataFrame`

        kwargs Possible: Non detections DataFrame.
        """
        raise NotImplementedError('SingleBandFeatureExtractor is an abstract class')

    def compute_by_bands(self, detections, bands=None,  **kwargs):
        if bands is None:
            bands = [1, 2]
        features_response = []
        for band in bands:
            features_response.append(self._compute_features(detections, band=band,  **kwargs))
        return pd.concat(features_response, axis=1)

    def get_features_keys(self, band):
        return [f'{x}_{band}' for x in self.features_keys]


# This is not being used.
# TODO delete this if it's not useful
#
# class FeatureExtractorFromFEList(FeatureExtractor):
#     def __init__(self, feature_extractor_list, bands):
#         self.feature_extractor_list = feature_extractor_list
#         self.bands = bands
#
#     def compute_features(self, detections, **kwargs):
#         """
#
#         Parameters
#         ----------
#         detections :class:pandas.`DataFrame`
#         kwargs Possible: Non detections DataFrame.
#
#         Returns :class:pandas.`DataFrame`
#         -------
#
#         """
#         if len(self.feature_extractor_list) == 0 or type(self.feature_extractor_list) is not list:
#             return pd.DataFrame()
#         if kwargs['non_detections'] is None:
#             print("non_detections not given")
#
#         non_detections = kwargs['non_detections']
#         all_features = pd.DataFrame(detections.index.unique('oid'))
#         all_features = all_features.set_index('oid')
#
#         for feature_extractor in self.feature_extractor_list:
#             if FeatureExtractorSingleBand in feature_extractor.__class__.__mro__:
#                 features = FeatureExtractorPerBandFromSingleLC(
#                     feature_extractor, self.bands).compute_features(detections)
#             else:
#                 features = feature_extractor.compute_features(detections, non_detections=non_detections)
#         all_features.join(features)
#         return all_features
