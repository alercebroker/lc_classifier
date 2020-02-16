import pandas as pd
from sklearn.model_selection import train_test_split
from late_classifier.classifier.preprocessing import FeaturePreprocessor
from late_classifier.classifier.preprocessing import intersect_oids_in_dataframes
from late_classifier.classifier.models import HierarchicalRandomForest
from late_classifier.classifier.metrics import confusion_matrix
from late_classifier.classifier.metrics import kaggle_score, classification_report


features = pd.read_pickle('dataset_features.pkl')
labels = pd.read_pickle('dataset_labels.pkl')


non_used_features = [
    'Mean_1',
    'Mean_2'
]
feature_preprocessor = FeaturePreprocessor(non_used_features)

features = feature_preprocessor.preprocess_features(features)
features, labels = intersect_oids_in_dataframes(features, labels)

train_val_oids, test_oids = train_test_split(
    labels.index.values, stratify=labels.classALeRCE.values, test_size=0.15)

train_oids, val_oids = train_test_split(
    train_val_oids, stratify=labels.loc[train_val_oids].classALeRCE.values,
    test_size=0.15/0.7)


taxonomy_dictionary = {
    'stochastic': ['LPV', 'QSO', 'YSO', 'CV/Nova', 'Blazar', 'AGN'],
    'periodic': ['RRL', 'EB', 'DSCT', 'Ceph', 'Periodic-Other'],
    'transient': ['SNIa', 'SNII', 'SNIbc', 'SLSN']
}
classifier = HierarchicalRandomForest(taxonomy_dictionary)

classifier.fit(features.loc[train_oids], labels.loc[train_oids])

test_predictions = classifier.predict(features.loc[test_oids])

test_confusion_matrix = confusion_matrix(
    test_predictions,
    labels.loc[test_oids],
    classifier.get_list_of_classes()
)

print(test_confusion_matrix)

test_predictions_proba = classifier.predict_proba(features.loc[test_oids])
kaggle_score_value = kaggle_score(test_predictions_proba, labels.loc[test_oids])
print(f'Kaggle score test: {kaggle_score_value}')

print(classification_report(test_predictions, labels.loc[test_oids]))
