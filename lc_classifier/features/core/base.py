import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List
import logging
from functools import lru_cache


class FeatureExtractor(ABC):
    @abstractmethod
    def get_features_keys(self) -> Tuple[str, ...]:
        raise NotImplementedError('get_features_keys is an abstract method')

    @abstractmethod
    def get_required_keys(self) -> Tuple[str, ...]:
        raise NotImplementedError('get_required_keys is an abstract method')

    def nan_df(self, index):
        """
        Method to generate a empty dataframe with NaNs.

        Parameters
        ---------
        index : str
            Name/id/oid of the observation

        Returns
        ------
        pd.DataFrame
            One-row dataframe full of nans.
        """
        return pd.DataFrame(
            columns=self.get_features_keys(), index=[index], dtype=np.float)

    def nan_series(self):
        return pd.Series(
            data=[np.nan]*len(self.get_features_keys()),
            index=self.get_features_keys())

    def compute_features(self, detections, **kwargs) -> pd.DataFrame:
        """
        Interface to implement different features extractors.

        Parameters
        ----------
        detections :
        DataFrame with detections of an object.

        kwargs Another arguments like Non detections.
        """
        logging.debug('base.py compute_features extractor %s type(detections) %s' % (
            self.__class__.__name__,
            type(detections)
        ))

        if len(detections) == 0:
            return pd.DataFrame(columns=self.get_features_keys(), index=[])

        if type(detections) == pd.core.groupby.generic.DataFrameGroupBy:
            aux_df = pd.DataFrame(columns=detections.obj.columns)
            if not self.has_all_columns(aux_df):
                raise Exception(
                    f'detections_df has missing columns: {self.__class__.__name__} requires {self.get_required_keys()}')
            features = self._compute_features_from_df_groupby(
                detections, **kwargs)
            return features

        elif type(detections) == pd.core.frame.DataFrame:
            if not self.has_all_columns(detections):
                raise Exception(
                    f'detections_df has missing columns: {self.__class__.__name__} requires {self.get_required_keys()}')
            features = self._compute_features(detections, **kwargs)
            return features

    @abstractmethod
    def _compute_features(self, detections: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError('_compute_features is an abstract method')

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        detections_ungrouped = detections.filter(lambda x: True)
        return self._compute_features(detections_ungrouped, **kwargs)

    def has_all_columns(self, df):
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


class FeatureExtractorSingleBand(FeatureExtractor, ABC):
    def __init__(self, bands: List):
        self.bands = bands

    def _compute_features(self, detections, **kwargs):
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

    def _compute_features_from_df_groupby(self, detections, **kwargs) -> pd.DataFrame:
        features_response = []
        for band in self.bands:
            features_response.append(
                self.compute_feature_in_one_band_from_group(detections, band, **kwargs))
        return pd.concat(features_response, axis=1)

    def compute_feature_in_one_band_from_group(self, detections, band, **kwargs):
        # Override this method to implement group processing.
        detections_ungrouped = detections.filter(lambda x: True)
        return self.compute_feature_in_one_band(detections_ungrouped, band, **kwargs)

    @abstractmethod
    def compute_feature_in_one_band(self, detections: pd.DataFrame, band, **kwargs) -> pd.DataFrame:
        """
        Computes features of single band detections from one object.

        Parameters
        ---------
        detections : pd.DataFrame
            Detections light curve from one object and one band.

        kwargs Possible: Non detections DataFrame.

        Returns
        ------
        pd.DataFrame
            Single-row dataframe with the computed features.
        """
        raise NotImplementedError(
            'compute_feature_in_one_band is an abstract method')

    def compute_by_bands(self, detections,  **kwargs):
        features_response = []
        for band in self.bands:
            features_response.append(
                self.compute_feature_in_one_band(detections, band=band, **kwargs))
        return pd.concat(features_response, axis=1, join="outer")

    @lru_cache(maxsize=10)
    def get_features_keys_with_band(self, band):
        return [f'{x}_{band}' for x in self.get_features_keys_without_band()]

    @abstractmethod
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        raise NotImplementedError(
            'get_features_keys_without_band is an abstract method')

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        feature_names = []
        for band in self.bands:
            feature_names += self.get_features_keys_with_band(band)
        return tuple(feature_names)

    def nan_series_in_band(self, band):
        columns = self.get_features_keys_with_band(band)
        return pd.Series(
            data=[np.nan]*len(columns),
            index=columns)


class FeatureExtractorComposer(FeatureExtractor):
    def __init__(self, feature_extractors: List[FeatureExtractor]):
        self.feature_extractors = feature_extractors

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        features_keys = []
        for extractor in self.feature_extractors:
            features_keys += extractor.get_features_keys()
        return tuple(features_keys)

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        required_keys = []
        for extractor in self.feature_extractors:
            required_keys += extractor.get_required_keys()
        return tuple(set(required_keys))

    def _compute_features(self, detections: pd.DataFrame, **kwargs) -> pd.DataFrame:
        feature_dataframes = []
        # TODO: loop over lightcurves instead of features in order to improve
        # cache friendliness. Requires heavy refactoring.
        for extractor in self.feature_extractors:
            feature_dataframes.append(
                extractor.compute_features(detections, **kwargs))
        return pd.concat(feature_dataframes, axis=1)
