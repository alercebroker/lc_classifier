import unittest
import pandas as pd

from late_classifier.classifier.models import HierarchicalRandomForest

TAXONOMY = {
    'Stochastic': ['LPV', 'QSO', 'YSO', 'CV/Nova', 'Blazar', 'AGN'],
    'Periodic': ['RRL', 'E', 'DSCT', 'CEP', 'Periodic-Other'],
    'Transient': ['SNIa', 'SNIbc', 'SLSN']
}


class TestHierarchicalRF(unittest.TestCase):
    def setUp(self) -> None:
        self.train_features = pd.read_csv('data_examples/2000_features.csv')
        self.train_labels = pd.read_csv('data_examples/2000_labels.csv')
        del self.train_features["oid"]

        self.test_features = pd.read_csv('data_examples/100_objects_features.csv')
        self.test_labels = pd.read_csv('data_examples/100_objects_labels.csv')
        del self.test_labels["objectId"]

        self.model_trained = HierarchicalRandomForest(TAXONOMY)
        self.model_trained.download_model()
        self.model_trained.load_model(self.model_trained.MODEL_PICKLE_PATH)

        self.model_silly = HierarchicalRandomForest(TAXONOMY)

    def test_predict(self):
        predicted_classes = self.model_trained.predict(self.test_features)
        self.assertListEqual(list(predicted_classes["classALeRCE"][-4:]),
                             list(self.test_labels["classALeRCE"][-4:])
                             )

    def test_fit(self):
        self.model_silly.fit(self.train_features, self.train_labels)
        probabilities = self.model_silly.predict_proba(self.train_features)
        predicted_classes = self.model_silly.predict(self.test_features)
        self.assertEqual(len(probabilities), 2000)
        self.assertEqual(len(predicted_classes), 98)
