from .base import GenericPreprocessor
import numpy as np
import pandas as pd


class ZTFLightcurvePreprocessor(GenericPreprocessor):
    def __init__(self, stream=False):
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
        self.stream = stream
        if not self.stream:
            self.not_null_columns.append('sgscore1')
        self.column_translation = {
            'mjd': 'time',
            'fid': 'band',
            'magpsf_ml': 'magnitude',
            'sigmapsf_ml': 'error'
        }
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
        Sometimes the same source triggers two detections with slightly
        different positions.

        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()

        # keep the one with best rb
        detections = detections.sort_values("rb", ascending=False)
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'mjd'], keep='first')
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
        def magpsf_ml_not_stream(detections, objects_table):
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

        def magpsf_ml_stream(detections):
            detections = detections.copy()
            is_corrected = detections.corrected.all()
            if is_corrected:
                detections["magpsf_ml"] = detections["magpsf_corr"]
                detections["sigmapsf_ml"] = detections["sigmapsf_corr_ext"]
            else:
                detections["magpsf_ml"] = detections["magpsf"]
                detections["sigmapsf_ml"] = detections["sigmapsf"]
            return detections

        grouped_detections = detections.groupby(level=0, sort=False, group_keys=False)
        if self.stream:
            detections = grouped_detections.apply(magpsf_ml_stream)
        else:
            detections = grouped_detections.apply(
                magpsf_ml_not_stream, objects_table=objects)
        return detections

    def preprocess(self, dataframe, objects=None):
        """
        :param dataframe:
        :param objects:
        :return:
        """
        if not self.stream and objects is None:
            raise Exception('ZTF Preprocessor requires objects dataframe')
        self.verify_dataframe(dataframe)
        dataframe = self.get_magpsf_ml(dataframe, objects)
        if not self.has_necessary_columns(dataframe):
            raise Exception('dataframe does not have all the necessary columns')
        dataframe = self.discard_bogus(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.enough_alerts(dataframe)
        dataframe = self.rename_columns_detections(dataframe)
        return dataframe

    def rename_columns_non_detections(self, non_detections):
        return non_detections.rename(
            columns=self.column_translation, errors='ignore')

    def rename_columns_detections(self, detections):
        return detections.rename(
            columns=self.column_translation, errors='ignore')


class ZTFForcedPhotometryLightcurvePreprocessor(GenericPreprocessor):
    """Preprocessing for lightcurves from ZTF forced photometry service."""
    def __init__(self):
        super().__init__()

        self.required_columns = [
            'mjd',
            'fid',
            'forcediffimflux',
            'forcediffimfluxunc',
            'mag_tot',
            'sigma_mag_tot',
        ]

        self.column_translation = {
            'mjd': 'time',
            'fid': 'band',
            'forcediffimflux': 'difference_flux',
            'forcediffimfluxunc': 'difference_flux_error',
            'mag_tot': 'magnitude',
            'sigma_mag_tot': 'error'
        }
        self.max_sigma = 1.0

        self.new_columns = []
        for c in self.required_columns:
            if c in self.column_translation.keys():
                self.new_columns.append(self.column_translation[c])
            else:
                self.new_columns.append(c)

        self.required_cols_metadata = ['ra', 'dec']

    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.required_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def metadata_has_necessary_columns(self, object_df):
        for c in self.required_cols_metadata:
            if c not in object_df.columns:
                return False
        return True

    def discard_invalid_value_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections.replace([np.inf, -np.inf], np.nan)
        detections[self.new_columns] = detections[self.new_columns].apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        return detections

    def drop_duplicates(self, detections):
        """
        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'time'])
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[
            ((detections['error'] > 0.0) &
             (detections['error'] < self.max_sigma)) | detections['error'].isna()
        ]
        return detections

    def enough_alerts(self, detections, min_dets=5):
        objects = detections.groupby("oid")
        indexes = []
        for oid, group in objects:
            if len(group.band == 1) > min_dets or len(group.band == 2) > min_dets:
                indexes.append(oid)
        return detections.loc[indexes]

    def discard_i_filter(self, detections):
        return detections[detections.band != 3]

    def preprocess(self, dataframe, objects=None):
        """
        :param dataframe:
        :param objects:
        :return:
        """
        self.verify_dataframe(dataframe)
        if not self.has_necessary_columns(dataframe):
            raise Exception(
                'Lightcurve dataframe does not have all the necessary columns')

        if not self.metadata_has_necessary_columns(objects):
            raise Exception(
                'Metadata dataframe does not have all the necessary columns')
        dataframe = self.rename_columns_detections(dataframe)
        dataframe = self.discard_i_filter(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.enough_alerts(dataframe)
        return dataframe

    def rename_columns_detections(self, detections):
        return detections.rename(
            columns=self.column_translation, errors='ignore')
