from .base import GenericPreprocessor
import numpy as np


class ElasticcPreprocessor(GenericPreprocessor):
    """Preprocessing for lightcurves from ZTF forced photometry service."""
    def __init__(self):
        super().__init__()

        self.required_columns = [
            'MJD',
            'FLUXCAL',
            'FLUXCALERR',
            'BAND'
        ]

        self.column_translation = {
            'MJD': 'time',
            'BAND': 'band',
            'FLUXCAL': 'difference_flux',
            'FLUXCALERR': 'difference_flux_error',
        }

        self.new_columns = []
        for c in self.required_columns:
            if c in self.column_translation.keys():
                self.new_columns.append(self.column_translation[c])
            else:
                self.new_columns.append(c)

    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.required_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[
            ((detections['difference_flux_error'] < 300)
             & (np.abs(detections['difference_flux']) < 50e3))]
        return detections

    def preprocess(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        self.verify_dataframe(dataframe)
        if not self.has_necessary_columns(dataframe):
            raise Exception(
                'Lightcurve dataframe does not have all the necessary columns')

        dataframe = self.rename_columns_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)

        # A little hack ;)
        dataframe['magnitude'] = dataframe['difference_flux']
        dataframe['error'] = dataframe['difference_flux_error']
        
        return dataframe

    def rename_columns_detections(self, detections):
        return detections.rename(
            columns=self.column_translation, errors='ignore')
