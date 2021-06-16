from abc import ABCMeta, abstractmethod
import pandas as pd


class GenericPreprocessor:
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    def verify_dataframe(self, dataframe):
        if type(dataframe) is not pd.DataFrame:
            raise ValueError("Input isn't a Pandas DataFrame")
        return

    @abstractmethod
    def preprocess(self, dataframe):
        """
        :param dataframe:
        """
        pass
