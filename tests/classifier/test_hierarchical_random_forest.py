import unittest
import pandas as pd

from lc_classifier.classifier.models import HierarchicalRandomForest


class TestHierarchicalRF(unittest.TestCase):
    def setUp(self) -> None:
        self.train_features = pd.read_csv('data_examples/2000_features.csv')
        self.train_labels = pd.read_csv('data_examples/2000_labels.csv')
        del self.train_features["objectId"]

        self.test_features = pd.read_csv('data_examples/100_objects_features.csv')
        self.test_labels = pd.read_csv('data_examples/100_objects_labels.csv')
        del self.test_labels["objectId"]

        self.model_trained = HierarchicalRandomForest()
        self.model_trained.download_model()
        self.model_trained.load_model(self.model_trained.MODEL_PICKLE_PATH)

        self.taxonomy = {
            'Stochastic': ['LPV', 'QSO', 'YSO', 'CV/Nova', 'Blazar', 'AGN'],
            'Periodic': ['RRL', 'E', 'DSCT', 'CEP', 'Periodic-Other'],
            'Transient': ['SNIa', 'SNIbc', 'SLSN']
        }
        self.model_silly = HierarchicalRandomForest(
            taxonomy_dictionary=self.taxonomy)

    def test_predict(self):
        predicted_classes = self.model_trained.predict(self.test_features)
        self.assertListEqual(
            list(predicted_classes["classALeRCE"][:5]),
            list(self.test_labels["classALeRCE"][:5])
        )

    def test_predict_in_pipeline(self):
        prediction = self.model_trained.predict_in_pipeline(
            self.test_features.iloc[[0]])
        self.assertIsInstance(prediction, dict)
        self.assertEqual(prediction["class"].iloc[0], "E")

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
