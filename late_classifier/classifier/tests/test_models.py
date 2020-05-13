import os
import unittest
import numpy as np
import pandas as pd
from late_classifier.classifier.models import BaselineRandomForest, HierarchicalRandomForest
from late_classifier.classifier.metrics import balanced_recall


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "data"))


class TestBaselineRF(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.read_pickle(
            os.path.join(EXAMPLES_PATH, 'some_features.pkl')
        )
        self.labels = pd.read_pickle(
            os.path.join(
                EXAMPLES_PATH, 'some_labels.pkl')
        )

    def test_fit(self):
        classifier = BaselineRandomForest()
        classifier.fit(self.features, self.labels)
        predicted_classes = classifier.predict(self.features)
        recall_values = balanced_recall(predicted_classes, self.labels)
        self.assertGreater(
            recall_values, 0.95,
            'Training balanced recall has to be greater than 95 %')

    def test_predict_proba(self):
        classifier = BaselineRandomForest()
        classifier.fit(self.features, self.labels)
        predicted_probs = classifier.predict_proba(self.features)
        probs_sum = predicted_probs.sum(axis=1)
        errors = np.abs(probs_sum - 1.0)
        self.assertTrue((errors < 1e-6).all())


class TestHierarchicalRF(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.read_pickle(
            os.path.join(EXAMPLES_PATH, 'some_features.pkl')
        )
        self.labels = pd.read_pickle(
            os.path.join(
                EXAMPLES_PATH, 'some_labels.pkl')
        )
        self.taxonomy_dictionary = {
            'stochastic': ['LPV', 'QSO', 'YSO', 'CV/Nova', 'Blazar', 'AGN'],
            'periodic': ['RRL', 'EB', 'DSCT', 'Ceph', 'Periodic-Other'],
            'transient': ['SNIa', 'SNII', 'SNIbc', 'SLSN']
        }

    def test_fit(self):
        classifier = HierarchicalRandomForest(self.taxonomy_dictionary)
        classifier.fit(self.features, self.labels)
        predicted_classes = classifier.predict(self.features)
        recall_values = balanced_recall(predicted_classes, self.labels)
        self.assertGreater(
            recall_values, 0.95,
            'Training balanced recall has to be greater than 95 %')

    def test_predict_proba(self):
        classifier = HierarchicalRandomForest(self.taxonomy_dictionary)
        classifier.fit(self.features, self.labels)
        predicted_probs = classifier.predict_proba(self.features)
        self.is_sum_one(predicted_probs)

    def test_smaller_dictionary(self):
        taxonomy_dictionary = self.taxonomy_dictionary
        taxonomy_dictionary['stochastic'] = taxonomy_dictionary['stochastic'][:-1]
        classifier = HierarchicalRandomForest(self.taxonomy_dictionary)
        with self.assertRaises(Exception):
            classifier.fit(self.features, self.labels)

    def test_larger_dictionary(self):
        taxonomy_dictionary = self.taxonomy_dictionary
        taxonomy_dictionary['stochastic'] = taxonomy_dictionary['stochastic']+['new class']
        classifier = HierarchicalRandomForest(self.taxonomy_dictionary)
        classifier.fit(self.features, self.labels)
        predicted_probs = classifier.predict_proba(self.features)
        self.assertEqual(predicted_probs.shape, (len(self.features), 15))

    def test_predict_in_pipeline(self):
        classifier = HierarchicalRandomForest(self.taxonomy_dictionary)
        classifier.fit(self.features, self.labels)
        predicted_classes = classifier.predict_in_pipeline(self.features)
        n_objects = len(self.features)
        self.is_sum_one(predicted_classes['probabilities'])
        self.assertEqual(n_objects, len(predicted_classes['probabilities']))
        self.is_sum_one(predicted_classes['hierarchical']['root'])
        self.assertEqual(n_objects, len(predicted_classes['hierarchical']['root']))
        for children_probs in predicted_classes['hierarchical']['children'].values():
            self.is_sum_one(children_probs)
            self.assertEqual(n_objects, len(children_probs))

    def is_sum_one(self, probs: pd.DataFrame):
        probs_sum = probs.sum(axis=1)
        errors = np.abs(probs_sum - 1.0)
        self.assertTrue((errors < 1e-6).all())


if __name__ == '__main__':
    unittest.main()
