import unittest
import pandas as pd
from lc_classifier.features import ZTFLightcurvePreprocessor


class PreprocessTest(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ZTFLightcurvePreprocessor()

        self.detections = pd.read_csv(
            "data_examples/100_objects_detections_corr.csv",
            index_col="objectId")
        self.detections.index.name = 'oid'

        self.objects = pd.read_csv(
            "data_examples/100_objects.csv", index_col="objectId")
        self.objects.index.name = 'oid'

    def test_preprocess(self):
        det_preprocess = self.preprocessor.preprocess(
            self.detections, objects=self.objects)
        self.assertEqual(5292, len(det_preprocess))
