import unittest
import numpy as np
import pandas as pd
from late_classifier.classifier.preprocessing import FeaturePreprocessor
from late_classifier.classifier.preprocessing import LabelPreprocessor


class TestFeaturePreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        self.features_sample = pd.read_pickle('features20191119.pkl')
        self.features_sample = self.features_sample.loc[
            ~self.features_sample.index.duplicated(keep='first')]

    def test_preprocess_features_nan(self):
        self.assertTrue(self.features_sample.isna().values.any())
        feature_preprocessor = FeaturePreprocessor()
        filtered_features = feature_preprocessor.preprocess_features(
            self.features_sample)
        self.assertFalse(filtered_features.isna().values.any())

    def test_preprocess_features_inf(self):
        df = pd.DataFrame([np.inf, 3])
        self.assertFalse(np.isfinite(df).all().all())
        feature_preprocessor = FeaturePreprocessor()
        filtered_df = feature_preprocessor.preprocess_features(
            df)
        self.assertTrue(np.isfinite(filtered_df).all().all())

    def test_if_losing_objects(self):
        n_objects_initial = len(self.features_sample)
        feature_preprocessor = FeaturePreprocessor()
        filtered_features = feature_preprocessor.preprocess_features(
            self.features_sample)
        self.assertTrue(len(filtered_features) == n_objects_initial)

    def test_non_used_features(self):
        feature_preprocessor = FeaturePreprocessor(
            non_used_features=['Mean_1'])
        self.assertTrue('Mean_1' in self.features_sample.columns)
        filtered_features = feature_preprocessor.preprocess_features(
            self.features_sample)
        self.assertFalse('Mean_1' in filtered_features.columns)


class TestLabelPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        self.label_processor = LabelPreprocessor('class_dictionary_test.json')

    def test_label_preprocessor(self):
        labels = pd.read_pickle('labels.pkl')
        original_classes = labels.classALeRCE.unique().tolist()
        self.assertTrue('ZZ' in original_classes)
        self.assertTrue('SNIIn' in original_classes)
        filtered_labels = self.label_processor.preprocess_labels(labels)
        filtered_classes = filtered_labels.classALeRCE.unique().tolist()
        self.assertFalse('ZZ' in filtered_classes)
        self.assertFalse('SNIIn' in filtered_classes)
