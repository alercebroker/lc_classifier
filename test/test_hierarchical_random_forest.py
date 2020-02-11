import unittest
import pandas as pd
import numpy as np

from late_classifier.classifier.models import HierarchicalRandomForest


class TestHierarchicalRF(unittest.TestCase):
    def setUp(self) -> None:
        self.train_features = pd.read_pickle('../paper_paula/train_features_hackathon.pkl')
        self.train_labels = pd.read_csv(
            '../paper_paula/train_labels_hackathon.csv',
            index_col='challenge_oid'
        )

    def test_fit(self):
        taxonomy_dictionary = {
            'stochastic': ['LPV', 'QSO', 'YSO', 'CV/Nova', 'Blazar', 'AGN'],
            'periodic': ['RRL', 'EB', 'DSCT', 'Ceph', 'Periodic-Other'],
            'transient': ['SNIa', 'SNII', 'SNIbc']
        }
        model = HierarchicalRandomForest(taxonomy_dictionary)
        model.fit(self.train_features, self.train_labels)
        probs = model.predict_proba(self.train_features)
        print(probs.head())
        predicted_classes = model.predict(self.train_features)
        print(predicted_classes.head())
