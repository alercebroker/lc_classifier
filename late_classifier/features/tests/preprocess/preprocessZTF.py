"""
Tests for preprocess ZTF data.
Author: Javier Arredondo @alercebroker
Date: 09/12/2019
"""

import unittest
import pandas as pd
from late_classifier.features import DetectionsPreprocessorZTF

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class TestObjectsMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor_ztf = DetectionsPreprocessorZTF()
        self.raw_data_ZTF18abakgtm = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_det.csv'), index_col="oid")
        self.raw_data_ZTF18abakgtm['sigmapsf_corr_ext'] = self.raw_data_ZTF18abakgtm['sigmapsf_corr']
        self.fake_objects = pd.DataFrame(
            columns=['corrected'],
            index=['ZTF18abakgtm'],
            data=[[True]]
        )

    def test_preprocess_one_object(self):
        self.assertEqual(self.preprocessor_ztf.has_necessary_columns(
            self.preprocessor_ztf.get_magpsf_ml(
                self.raw_data_ZTF18abakgtm,
                self.fake_objects
            )),
            True)
        self.assertEqual(self.raw_data_ZTF18abakgtm.index.name, "oid")
        preprocessed_data = self.preprocessor_ztf.preprocess(
            self.raw_data_ZTF18abakgtm, objects=self.fake_objects)
        self.assertEqual(type(preprocessed_data), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()

