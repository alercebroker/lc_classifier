import pandas as pd
from sklearn.model_selection import train_test_split
from lc_classifier.augmentation.utils import augment_just_training_set_features
from lc_classifier.classifier.preprocessing import FeaturePreprocessor
from lc_classifier.classifier.preprocessing import intersect_oids_in_dataframes
from lc_classifier.classifier.models import HierarchicalRandomForest
from lc_classifier.classifier.metrics import confusion_matrix
from lc_classifier.classifier.metrics import kaggle_score, classification_report


features = pd.read_pickle('dataset_features_with_wise.pkl')
labels = pd.read_pickle('dataset_labels.pkl')
labels = labels[['classALeRCE']]

non_used_features = [
    'Mean_1',
    'Mean_2'
]
feature_preprocessor = FeaturePreprocessor(non_used_features)

features = feature_preprocessor.preprocess_features(features)
features, labels = intersect_oids_in_dataframes(features, labels)

train_val_oids, test_oids = train_test_split(
    labels.index.values, stratify=labels.classALeRCE.values, test_size=0.15, random_state=0)

train_oids, val_oids = train_test_split(
    train_val_oids, stratify=labels.loc[train_val_oids].classALeRCE.values,
    test_size=0.15/0.7, random_state=0)

# needed for color augmentation
training_features = features.loc[train_oids]
training_labels = labels.loc[train_oids]
training_features.to_pickle('training_features.pkl')
training_labels.to_pickle('training_labels.pkl')

taxonomy_dictionary = {
    'Stochastic': ['LPV', 'QSO', 'YSO', 'CV/Nova', 'Blazar', 'AGN'],
    'Periodic': ['RRL', 'EB', 'DSCT', 'Ceph', 'Periodic-Other'],
    'Transient': ['SNIa', 'SNII', 'SNIbc', 'SLSN']
}
classifier = HierarchicalRandomForest(taxonomy_dictionary)

classifier.fit(features.loc[train_oids], labels.loc[train_oids])


def evaluate_classifier(classifier):
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


evaluate_classifier(classifier)
## AUGMENTATION

gp_features = pd.read_pickle(
    '~/alercebroker/GP-Augmentation/results_paula/augmented_features_with_colors.pkl')

gp_labels = pd.read_pickle(
    '~/alercebroker/GP-Augmentation/results_paula/augmented_labels.pkl')

gp_features = feature_preprocessor.preprocess_features(gp_features)
gp_features, gp_labels = intersect_oids_in_dataframes(gp_features, gp_labels)

# subsample to have 50 % original + 50 % augmented
subsampled_aug_oids = []
for astro_class in gp_labels.classALeRCE.unique():
    n_training_samples = len(training_labels[training_labels.classALeRCE == astro_class])
    class_df = gp_labels[gp_labels.classALeRCE == astro_class]
    subsampled_df = class_df.sample(
        min(n_training_samples, len(class_df)), random_state=0)
    subsampled_aug_oids += subsampled_df.index.values.tolist()

gp_labels = gp_labels.loc[subsampled_aug_oids]
gp_features = gp_features.loc[subsampled_aug_oids]

# gp_labels = gp_labels[gp_labels.classALeRCE.isin(['SNIa', 'SNII'])]
# gp_features = gp_features.loc[gp_labels.index]

aug_train_features, aug_train_labels = augment_just_training_set_features(
    features.loc[train_oids],
    labels.loc[train_oids],
    gp_features,
    gp_labels)
    
augmented_classifier = HierarchicalRandomForest(taxonomy_dictionary)
augmented_classifier.fit(aug_train_features, aug_train_labels)
evaluate_classifier(augmented_classifier)
