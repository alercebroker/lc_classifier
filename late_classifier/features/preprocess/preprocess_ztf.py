from .base import GenericPreprocessor
from functools import reduce
import numpy as np
import pandas as pd


class DetectionsPreprocessorZTF(GenericPreprocessor):
    def __init__(self):
        super().__init__()
        self.necessary_columns = ['mjd', 'ra', 'dec', 'magpsf_corr', 'sigmapsf_corr', 'rb', 'sgscore1']
        self.max_sigma = 1.0
        self.rb_threshold = 0.55

    def valid_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        booleans = list(map(lambda x: x in dataframe.columns, self.necessary_columns))
        return reduce(lambda x, y: x & y, booleans)

    def discard_invalid_value_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections.replace([np.inf, -np.inf], np.nan)
        valid_alerts = detections[self.necessary_columns].notna().all(axis=1)
        detections = detections[valid_alerts.values]
        detections[self.necessary_columns] = detections[self.necessary_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        return detections

    def drop_duplicates(self, detections):
        """
        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'mjd'])
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[
            (detections.sigmapsf_corr > 0.0)
            & (detections.sigmapsf_corr < self.max_sigma)]
        return detections

    def discard_bogus(self, detections):
        """

        :param detections:
        :return:
        """
        detections = detections[detections.rb >= self.rb_threshold]
        return detections

    def preprocess(self, dataframe):
        """

        :param dataframe:
        :return:
        """
        self.verify_dataframe(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.discard_bogus(dataframe)
        return dataframe


class DetectionsV2PreprocessorZTF(DetectionsPreprocessorZTF):
    def preprocess(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        return dataframe
