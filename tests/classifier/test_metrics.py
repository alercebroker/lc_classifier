import unittest
import numpy as np
import pandas as pd

from lc_classifier.classifier.metrics import kaggle_score, balanced_recall, confusion_matrix, classification_report
from lc_classifier.classifier.models import HierarchicalRandomForest


class KaggleScoreTest(unittest.TestCase):
    def test_all_label_classes_in_prediction(self):
        labels = pd.DataFrame(
            np.array(['class_1', 'class_2', 'class_3']).reshape(-1, 1),
            columns=['classALeRCE'],
            index=['ZTF18asdf', 'ZTF19qwerty', 'ZTF20poiu']
        )

        predictions = pd.DataFrame(
            np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]),
            columns=['class_1', 'class_2'],
            index=['ZTF18asdf', 'ZTF19qwerty', 'ZTF20poiu']
        )

        self.assertRaises(Exception, kaggle_score, predictions, labels)


class HRFMetricsTest(unittest.TestCase):
    def setUp(self):
        self.model_trained = HierarchicalRandomForest()
        self.model_trained.download_model()
        self.model_trained.load_model(self.model_trained.MODEL_PICKLE_PATH)

        self.train_features = pd.read_csv('data_examples/2000_features.csv')
        self.train_labels = pd.read_csv('data_examples/2000_labels.csv')
        del self.train_features["objectId"]

        self.predicted = self.model_trained.predict(self.train_features)
        self.probabilities = self.model_trained.predict_proba(self.train_features)

    def test_perfect_confusion_matrix(self):
        cf = confusion_matrix(self.train_labels, self.train_labels)
        diagonal = np.diag(cf).sum()
        self.assertEqual(diagonal, len(self.train_labels))

    def test_model_confusion_matrix(self):
        cf = confusion_matrix(self.train_labels, self.predicted)
        diagonal = np.diag(cf).sum()
        self.assertGreaterEqual(diagonal, 1730)

    def test_model_balanced_recall(self):
        br = balanced_recall(self.predicted, self.train_labels)
        self.assertAlmostEqual(br, 0.81, 2)

    def test_model_classification_report(self):
        cr = classification_report(self.train_labels, self.predicted)
        self.assertEqual(str, type(cr))
        accuracy = cr.split()[-14]
        self.assertEqual(accuracy, "0.87")

    def test_model_kaggle_score(self):
        ks = kaggle_score(self.probabilities, self.train_labels)
        self.assertAlmostEqual(ks, 1., 0)
