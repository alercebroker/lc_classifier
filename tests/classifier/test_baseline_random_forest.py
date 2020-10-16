import unittest
import pandas as pd

from lc_classifier.classifier.models import BaselineRandomForest


class TestBaselineRF(unittest.TestCase):
    def setUp(self) -> None:
        self.train_features = pd.read_csv('data_examples/2000_features.csv')
        self.train_labels = pd.read_csv('data_examples/2000_labels.csv')
        del self.train_features["objectId"]

        self.test_features = pd.read_csv('data_examples/100_objects_features.csv')
        self.test_labels = pd.read_csv('data_examples/100_objects_labels.csv')
        del self.test_features["objectId"]

        self.model_silly = BaselineRandomForest()

    def test_fit(self):
        self.model_silly.fit(self.train_features, self.train_labels)
        probabilities = self.model_silly.predict_proba(self.train_features)
        predicted_classes = self.model_silly.predict(self.test_features)
        self.assertEqual(len(probabilities), 2000)
        self.assertEqual(len(predicted_classes), 98)

    def test_io(self):
        self.model_silly.fit(self.train_features, self.train_labels)
        self.model_silly.save_model("")
        self.model_silly.load_model("")

    def test_classes(self):
        self.model_silly.fit(self.train_features, self.train_labels)
        predicted_class = list(self.model_silly.get_list_of_classes())
        trained_class = list(self.train_labels["classALeRCE"].unique())
        self.assertListEqual(sorted(predicted_class), sorted(trained_class))
