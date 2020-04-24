import pandas as pd
from abc import ABC, abstractmethod
from typing import List


class FeatureExtractor(ABC):
    @abstractmethod
    def get_features_keys(self) -> List[str]:
        raise NotImplementedError('get_features_keys is an abstract method')

    @abstractmethod
    def get_required_keys(self) -> List[str]:
        raise NotImplementedError('get_required_keys is an abstract method')

    def validate_df(self, df):
        """
        Method to validate input detections. The required keys must be in df columns.

        Parameters
        ----------
        df :class:pandas.`DataFrame`
        DataFrame with detections of an object.
        """
        cols = set(df.columns)
        required = set(self.get_required_keys())
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
        return pd.DataFrame(columns=self.get_features_keys(), index=[index])

    @abstractmethod
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


class FeatureExtractorSingleBand(FeatureExtractor, ABC):
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

    @abstractmethod
    def compute_feature_in_one_band(self, detections: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Interface to compute features to single band detections of an object.
        Parameters
        ----------
        detections :class:pandas.`DataFrame`

        kwargs Possible: Non detections DataFrame.
        """
        raise NotImplementedError('compute_feature_in_one_band is an abstract class')

    def compute_by_bands(self, detections, bands=None,  **kwargs):
        if bands is None:
            bands = [1, 2]
        features_response = []
        for band in bands:
            features_response.append(self.compute_feature_in_one_band(detections, band=band, **kwargs))
        return pd.concat(features_response, axis=1)

    def get_features_keys_with_band(self, band):
        return [f'{x}_{band}' for x in self.get_features_keys()]
