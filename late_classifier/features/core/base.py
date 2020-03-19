from .decorators import *
import pandas as pd


class FeatureExtractor:
    def compute_features(self, detections, **kwargs):
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
        n_bands = len(detections.fid.unique())
        if n_bands > 1:
            raise Exception('SingleBandFeatureExtractor cannot handle multiple bands')
        return self._compute_features(detections, **kwargs)

    def _compute_features(self, detections, **kwargs):
        """
        Interface to compute features to single band detections of an object.
        Parameters
        ----------
        detections :class:pandas.`DataFrame`

        kwargs Possible: Non detections DataFrame.
        """
        raise NotImplementedError('SingleBandFeatureExtractor is an abstract class')


class FeatureExtractorFromFEList(FeatureExtractor):
    def __init__(self, feature_extractor_list, bands):
        self.feature_extractor_list = feature_extractor_list
        self.bands = bands

    def compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        kwargs Possible: Non detections DataFrame.

        Returns :class:pandas.`DataFrame`
        -------

        """
        if len(self.feature_extractor_list) == 0 or type(self.feature_extractor_list) is not list:
            return pd.DataFrame()
        if kwargs['non_detections'] is None:
            print("non_detections not given")

        non_detections = kwargs['non_detections']
        all_features = pd.DataFrame(detections.index.unique('oid'))
        all_features = all_features.set_index('oid')

        for feature_extractor in self.feature_extractor_list:
            if FeatureExtractorSingleBand in feature_extractor.__class__.__mro__:
                features = FeatureExtractorPerBandFromSingleLC(feature_extractor, self.bands).compute_features(detections)
            else:
                features = feature_extractor.compute_features(detections, non_detections=non_detections)
        all_features.join(features)
        return all_features
