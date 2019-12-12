import unittest
import numpy as np
import pandas as pd
from late_classifier.classifier.metrics import kaggle_score


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
