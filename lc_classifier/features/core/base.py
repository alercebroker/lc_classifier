import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
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
        return pd.DataFrame(columns=self.get_features_keys(), index=[index])

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
                oids = detections.obj.index.unique()
                logging.info(
                    f'detections_df has missing columns: {self.__class__.__name__} requires {self.get_required_keys()}')
                return self.nan_df(oids)
            features = self._compute_features_from_df_groupby(
                detections, **kwargs)
            return features

        elif type(detections) == pd.core.frame.DataFrame:
            if not self.has_all_columns(detections):
                oids = detections.index.unique()
                logging.info(
                    f'detections_df has missing columns: {self.__class__.__name__} requires {self.get_required_keys()}')
                return self.nan_df(oids)
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

    def _compute_features_from_df_groupby(self, detections, bands=None, **kwargs) -> pd.DataFrame:
        if bands is None:
            bands = [1, 2]
        features_response = []
        for band in bands:
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
            'compute_feature_in_one_band is an abstract class')

    def compute_by_bands(self, detections, bands=None,  **kwargs):
        if bands is None:
            bands = [1, 2]
        features_response = []
        for band in bands:
            features_response.append(
                self.compute_feature_in_one_band(detections, band=band, **kwargs))
        return pd.concat(features_response, axis=1, join="outer")

    @lru_cache(maxsize=10)
    def get_features_keys_with_band(self, band):
        return [f'{x}_{band}' for x in self.get_features_keys()]

    def nan_series_in_band(self, band):
        columns = self.get_features_keys_with_band(band)
        return pd.Series(
            data=[np.nan]*len(columns),
            index=columns)
