import unittest
import late_classifier.features.extractors as extractors
import pandas as pd


class WiseStreamExtractorTest(unittest.TestCase):
    def setUp(self):
        self.extractor = extractors.WiseStreamExtractor()
        self.detections = pd.DataFrame(
            {"fid": [1, 1, 2, 2], "sigmapsf_corr": [2, 2, 3, 3]}
        )
        self.xmatch = {"W1mag": 1.0, "W2mag": 1.0, "W3mag": 1.0}

    def test_get_features_keys(self):
        features = self.extractor.get_features_keys()
        self.assertEqual(features, ["W1-W2", "W2-W3", "g-W2", "g-W3", "r-W2", "r-W3"])

    def test_calculate_bands(self):
        g, r = self.extractor.calculate_bands(self.detections)
        self.assertEqual(g, 2)
        self.assertEqual(r, 3)

    def test_compute_features_xmatch(self):
        colors = self.extractor.compute_features(self.detections, xmatches=self.xmatch)
        self.assertIsInstance(colors, pd.DataFrame)
