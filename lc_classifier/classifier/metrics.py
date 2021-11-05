import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import classification_report as sklearn_classification_report


def balanced_recall(predictions, labels):
    """Computes the balanced recall between the predictions from a model
     and the real labels.

     Parameters
     ---------
     predictions : pd.DataFrame
        Predicted classes for multiple objects.
        Must contain a 'classALeRCE' column.
     labels : pd.DataFrame
        True labels for multiple objects.
        Must contain a 'classALeRCE' column.

     Returns
     ------
     np.float
     """
    recalls = []
    for classALeRCE in labels.classALeRCE.unique():
        labels_from_class = labels[labels.classALeRCE == classALeRCE]
        predictions_from_class = predictions.loc[labels_from_class.index]
        tp = predictions_from_class[predictions_from_class.classALeRCE == classALeRCE]
        recall = len(tp) / len(predictions_from_class)
        recalls.append(recall)
    return np.array(recalls).mean()


def confusion_matrix(predictions, labels, columns=None):
    if columns is None:
        columns = predictions.classALeRCE.unique()
    matrix = sklearn_confusion_matrix(
        labels.classALeRCE,
        predictions.classALeRCE,
        labels=columns)
    confusion_matrix_df = pd.DataFrame(
        matrix,
        columns=columns,
        index=columns
    )
    return confusion_matrix_df


def kaggle_score(prediction_probs, labels):
    astro_classes = labels.classALeRCE.unique().tolist()
    if not set(astro_classes).issubset(
            set(prediction_probs.columns.values.tolist())):
        raise Exception('There are classes in labels that are not present in the predictions. Cannot compute kaggle '
                        'score')
    scores = []
    for astro_class in astro_classes:
        class_labels = labels[labels.classALeRCE == astro_class]
        class_oids = class_labels.index.values
        class_probs = prediction_probs[astro_class].loc[class_oids].values.flatten()
        class_probs = np.clip(class_probs, 10**-15, 1-10**-15)
        class_score = -np.mean(np.log(class_probs))
        scores.append(class_score)
    return np.array(scores).mean()


def classification_report(predictions, labels, output_dict=False):
    if not predictions.index.equals(labels.index):
        raise Exception('Objects on both dataframes should match')
    predictions = predictions.loc[labels.index]
    return sklearn_classification_report(
        labels.classALeRCE, predictions.classALeRCE, output_dict=output_dict)
