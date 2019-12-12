import unittest
import pandas as pd
import numpy as np
from late_classifier.classifier.preprocessing import LabelPreprocessor
from late_classifier.classifier.models import BaselineRandomForest
from late_classifier.classifier.metrics import balanced_recall, kaggle_score
# from baseline_classifier.metrics import confusion_matrix


class TestBaselineRF(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_random_forest = BaselineRandomForest()
        features = pd.read_pickle('features20191119.pkl')
        features = features.loc[~features.index.duplicated(keep='first')]
        labels = pd.read_pickle('labels.pkl')
        label_preprocessor = LabelPreprocessor('class_dictionary_test.json')
        labels = label_preprocessor.preprocess_labels(labels)
        oid_features = set(features.index.values.tolist())
        oid_labels = set(labels.index.values.tolist())
        oids = oid_features.intersection(oid_labels)
        self.x_train = features.loc[oids]
        self.y_train = labels.loc[oids]
        self.assertTrue(len(self.x_train) == len(self.y_train))

    def test_fit(self):
        self.baseline_random_forest.fit(self.x_train, self.y_train)
        train_predictions = self.baseline_random_forest.predict(
            self.x_train[np.random.permutation(self.x_train.columns)])
        self.assertTrue(len(self.x_train) == len(train_predictions))
        balanced_recall_value = balanced_recall(train_predictions, self.y_train)
        # print(confusion_matrix(train_predictions, self.y_train))
        self.assertLessEqual(1.0 - balanced_recall_value, 0.4)

    def test_predict_proba(self):
        self.baseline_random_forest.fit(self.x_train, self.y_train)
        predicted_probs = self.baseline_random_forest.predict_proba(self.x_train)
        predicted_probs_classes = set(predicted_probs.columns.tolist())
        rf_classes = set(self.baseline_random_forest.get_list_of_classes())
        self.assertTrue(predicted_probs_classes == rf_classes)

        self.assertTrue(
            all(np.abs(predicted_probs.values.sum(axis=1) - 1.0) < 10e-3))
        kaggle_score_value = kaggle_score(predicted_probs, self.y_train)
        self.assertLessEqual(kaggle_score_value, 0.8 * np.log(len(predicted_probs_classes)))
