"""
Tests for preprocess ZTF data.
Author: Javier Arredondo @alercebroker
Date: 09/12/2019
"""

import unittest
import pandas as pd
import late_classifier.features.preprocess.preprocess_ztf as pztf

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class TestObjectsMethods(unittest.TestCase):
    preprocessor_ztf = pztf.DetectionsPreprocessorZTF()
    raw_data_ZTF18abakgtm = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_det.csv'), index_col="oid")

    def test_preprocess_one_object(self):
        self.assertEqual(self.preprocessor_ztf.valid_columns(self.raw_data_ZTF18abakgtm), True)
        self.assertEqual(self.raw_data_ZTF18abakgtm.index.name, "oid")
        preprocessed_data = self.preprocessor_ztf.preprocess(self.raw_data_ZTF18abakgtm)
        self.assertEqual(type(preprocessed_data), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()

