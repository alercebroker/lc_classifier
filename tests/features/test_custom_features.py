import unittest
import pandas as pd
from lc_classifier.features import CustomHierarchicalExtractor, CustomStreamHierarchicalExtractor


class CustomHierarchicalExtractorTest(unittest.TestCase):
    def setUp(self):
        self.custom_hierarchical_features = CustomHierarchicalExtractor()
        self.custom_hierarchical_features_stream = CustomStreamHierarchicalExtractor()

        self.detections = pd.read_csv(
            "data_examples/100_objects_detections_corr.csv", index_col="objectId")
        self.detections.index.name = 'oid'

        self.non_detections = pd.read_csv(
            "data_examples/100_objects_non_detections.csv", index_col="objectId")
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

    def test_custom_hierarchical_features_stream(self):
        oid = "ZTF17aaaorfd"
        detections = self.detections.loc[[oid]]
        non_detections = self.non_detections.loc[[oid]]
        xmatches = pd.DataFrame(
            data=[[1.0, 1.0, 1.0, 'ZTF17aaaorfd']],
            columns=['W1mag', 'W2mag', 'W3mag', 'oid'])
        
        metadata = pd.DataFrame(
            data=[['ZTF17aaaorfd', 'fake_candid', 1.0]],
            columns=['oid', 'candid', 'sgscore1'])
        
        features_df = self.custom_hierarchical_features_stream.compute_features(
            detections=detections,
            non_detections=non_detections,
            xmatches=xmatches,
            metadata=metadata)
        self.assertEqual(len(features_df), 1)

    def test_get_features_keys(self):
        keys = self.custom_hierarchical_features.get_features_keys()
        keys_stream = self.custom_hierarchical_features_stream.get_features_keys()
        self.assertEqual(len(keys), 22)
        self.assertEqual(len(keys_stream), 23)
