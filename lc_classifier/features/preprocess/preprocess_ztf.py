from .base import GenericPreprocessor
import numpy as np
import pandas as pd


class DetectionsPreprocessorZTF(GenericPreprocessor):
    def __init__(self):
        super().__init__()
        self.not_null_columns = [
            'mjd',
            'fid',
            'magpsf',
            'sigmapsf',
            'magpsf_ml',
            'sigmapsf_ml',
            'ra',
            'dec',
            'rb'
        ]
        self.max_sigma = 1.0
        self.rb_threshold = 0.55

    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.not_null_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def discard_invalid_value_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections.replace([np.inf, -np.inf], np.nan)
        valid_alerts = detections[self.not_null_columns].notna().all(axis=1)
        detections = detections[valid_alerts.values]
        detections[self.not_null_columns] = detections[self.not_null_columns].apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        return detections

    def drop_duplicates(self, detections):
        """
        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()
        detections = detections.sort_values("mjd", ascending=True)
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'mjd'])
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[((detections['sigmapsf_ml'] > 0.0) &
                                 (detections['sigmapsf_ml'] < self.max_sigma))
                                ]
        return detections

    def discard_bogus(self, detections):
        """

        :param detections:
        :return:
        """
        detections = detections[detections['rb'] >= self.rb_threshold]
        return detections

    def enough_alerts(self, detections, min_dets=5):
        objects = detections.groupby("oid")
        indexes = []
        for oid, group in objects:
            if len(group.fid == 1) > min_dets or len(group.fid == 2) > min_dets:
                indexes.append(oid)
        return detections.loc[indexes]

    def get_magpsf_ml(self, detections, objects):
        def magpsf_ml(detections, objects_table):
            detections = detections.copy()
            oid = detections.index.values[0]
            is_corrected = objects_table.loc[[oid]].corrected.values[0]
            if is_corrected:
                detections["magpsf_ml"] = detections["magpsf_corr"]
                detections["sigmapsf_ml"] = detections["sigmapsf_corr_ext"]
            else:
                detections["magpsf_ml"] = detections["magpsf"]
                detections["sigmapsf_ml"] = detections["sigmapsf"]
            return detections

        detections = detections.groupby(level=0, sort=False)\
            .apply(magpsf_ml, objects_table=objects).droplevel(level=1)
        return detections

    def preprocess(self, dataframe, objects=None):
        """
        :param dataframe:
        :param objects:
        :return:
        """
        if objects is None:
            raise Exception('ZTF Preprocessor requires objects dataframe')
        self.verify_dataframe(dataframe)
        dataframe = self.get_magpsf_ml(dataframe, objects)
        if not self.has_necessary_columns(dataframe):
            raise Exception('dataframe does not have all the necessary columns')
        dataframe.sort_values("mjd", inplace=True)
        dataframe = self.discard_bogus(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.enough_alerts(dataframe)
        return dataframe
