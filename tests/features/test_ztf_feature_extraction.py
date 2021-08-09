import unittest
import pandas as pd
from lc_classifier.features import ZTFFeatureExtractor, ZTFLightcurvePreprocessor


class ZTFFeatureExtractorTest(unittest.TestCase):
    def setUp(self):
        self.detections = pd.read_csv(
            "data_examples/100_objects_detections_corr.csv", index_col="objectId")
        self.detections.index.name = 'oid'

        self.non_detections = pd.read_csv(
            "data_examples/100_objects_non_detections.csv", index_col="objectId")
        self.non_detections.index.name = 'oid'
        self.non_detections["mjd"] = self.non_detections.jd - 2400000.5

        self.objects = pd.read_csv("data_examples/100_objects.csv", index_col="objectId")
        self.objects.index.name = 'oid'

    def test_feature_extraction(self):
        feature_extractor = ZTFFeatureExtractor(
            bands=(1, 2), stream=False)
        preprocessor = ZTFLightcurvePreprocessor(stream=False)

        detections = preprocessor.preprocess(
            self.detections, objects=self.objects)

        non_detections = preprocessor.rename_columns_non_detections(
            self.non_detections)

        features_df = feature_extractor.compute_features(
            detections=detections,
            non_detections=non_detections)
        self.assertEqual(98, features_df.shape[0])
        self.assertEqual(178, features_df.shape[1])
        for f in features_df.columns:
            print(f)

    def test_feature_extraction_stream(self):
        oid = "ZTF17aaaorfd"
        detections = self.detections.loc[[oid]]
        detections['corrected'] = True
        non_detections = self.non_detections.loc[[oid]]
        xmatches = pd.DataFrame(
            data=[[1.0, 1.0, 1.0, 'ZTF17aaaorfd']],
            columns=['W1mag', 'W2mag', 'W3mag', 'oid'])
        
        metadata = pd.DataFrame(
            data=[['ZTF17aaaorfd', 'fake_candid', 1.0]],
            columns=['oid', 'candid', 'sgscore1'])

        feature_extractor = ZTFFeatureExtractor(
            bands=(1, 2), stream=True)
        preprocessor = ZTFLightcurvePreprocessor(stream=True)

        detections = preprocessor.preprocess(detections)

        non_detections = preprocessor.rename_columns_non_detections(
            non_detections)

        features_df = feature_extractor.compute_features(
            detections=detections,
            non_detections=non_detections,
            xmatches=xmatches,
            metadata=metadata)
        self.assertEqual(len(features_df), 1)

    def test_get_features_keys(self):
        keys = ZTFFeatureExtractor(
            bands=(1, 2), stream=False).get_features_keys()
        keys_stream = ZTFFeatureExtractor(
            bands=(1, 2), stream=True).get_features_keys()
        self.assertEqual(178, len(keys))
        self.assertEqual(178, len(keys_stream))
