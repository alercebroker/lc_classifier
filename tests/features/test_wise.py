import unittest
import pandas as pd
import lc_classifier.features.extractors as extractors
from lc_classifier.features import ZTFLightcurvePreprocessor


class WiseStreamExtractorTest(unittest.TestCase):
    def setUp(self):
        self.extractor = extractors.WiseStreamExtractor()
        self.detections = pd.DataFrame(
            {"fid": [1, 1, 2, 2], "magpsf_ml": [2, 2, 3, 3]},
            index=['ZTF00test']*4
        )
        self.xmatch = pd.DataFrame(
            data=[[1.0, 1.0, 1.0, 'ZTF00test']],
            columns=['W1mag', 'W2mag', 'W3mag', 'oid'])
        self.preprocessor = ZTFLightcurvePreprocessor()
        self.detections = self.preprocessor.rename_columns_detections(
            self.detections)

    def test_get_features_keys(self):
        features = self.extractor.get_features_keys()
        self.assertEqual(
            features,
            ("W1-W2", "W2-W3", "g-W2", "g-W3", "r-W2", "r-W3"))

    def test_calculate_bands(self):
        g, r = self.extractor.calculate_bands(self.detections)
        self.assertEqual(g.iloc[0], 2)
        self.assertEqual(r.iloc[0], 3)

    def test_compute_features_xmatch(self):
        colors = self.extractor.compute_features(self.detections, xmatches=self.xmatch)
        self.assertIsInstance(colors, pd.DataFrame)


class WiseExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = extractors.WiseStaticExtractor()

        self.detections = pd.read_csv("data_examples/100_objects_detections_corr.csv", index_col="objectId")
        self.detections.index.name = 'oid'

        self.objects = pd.read_csv("data_examples/100_objects.csv", index_col="objectId")
        self.objects.index.name = 'oid'

        self.detections = ZTFLightcurvePreprocessor().get_magpsf_ml(
            detections=self.detections,
            objects=self.objects
        )

    def test_many_objects(self):
        wise_colors = self.extractor.compute_features(self.detections)
        self.assertEqual(
            wise_colors.shape,
            (98, len(self.extractor.get_features_keys())))
