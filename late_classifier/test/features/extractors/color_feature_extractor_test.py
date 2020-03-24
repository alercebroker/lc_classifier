import unittest
from .generic_extractor_test import GenericExtractorTest
from late_classifier.features.extractors.color_feature_extractor import ColorFeatureExtractor


class ColorFeatureExtractorTest(GenericExtractorTest, unittest.TestCase):
    extractor = ColorFeatureExtractor
    params = {
        "TEST_DETECTION": "data_example/detections_1.csv",
    }
