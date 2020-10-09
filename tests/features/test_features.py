import unittest
import pandas as pd
from late_classifier.features import CustomHierarchicalExtractor


class FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.custom_hierarchical_features = CustomHierarchicalExtractor()

        self.detections = pd.read_csv("data_examples/100_objects_detections_corr.csv", index_col="objectId")
        self.detections.index.name = 'oid'

        self.non_detections = pd.read_csv("data_examples/100_objects_non_detections.csv", index_col="objectId")
        self.non_detections.index.name = 'oid'
        self.non_detections["mjd"] = self.non_detections.jd - 2400000.5

        self.objects = pd.read_csv("data_examples/100_objects.csv", index_col="objectId")
        self.objects.index.name = 'oid'

    def test_custom_hierarchical_features(self):
        features_df = self.custom_hierarchical_features.compute_features(
            detections=self.detections,
            non_detections=self.non_detections,
            objects=self.objects
        )
        self.assertEqual(features_df.shape[0], 98)
        self.assertEqual(features_df.shape[1], 172)


