import pandas as pd

from lc_classifier.features import ZTFLightcurvePreprocessor


# Loading data
detections = pd.read_csv(
    'detections.csv',
    index_col='oid',
    na_values=['Infinity'])

non_detections = pd.read_csv(
    'non_detections.csv',
    index_col='oid',
    na_values=['Infinity'])

labels = pd.read_csv(
    'dfcrossmatches_prioritized_v5.1.csv',
    index_col='oid',
    na_values=['Infinity']
)

# Adapt classes for the paper
unused_classes = ['TDE', 'ZZ']
rename_class_dictionary = {
    'EA': 'EB',
    'EB/EW': 'EB',
    'RSCVn': 'Periodic-Other',
    'SNIIb': 'SNII',
    'SNIIn': 'SNII'
}

labels = labels[~labels.classALeRCE.isin(unused_classes)].copy()
labels['classALeRCE'] = labels['classALeRCE'].map(
    rename_class_dictionary).fillna(labels['classALeRCE'])

# Intersecting labels and detections
valid_oids = detections.index.unique().intersection(
    labels.index.unique())

labeled_detections = detections.loc[valid_oids]
labels = labels.loc[valid_oids].copy()

valid_oids = valid_oids.intersection(non_detections.index.unique())
labeled_non_detections = non_detections.loc[valid_oids]

# ZTF preprocessing
preprocessor_ztf = ZTFLightcurvePreprocessor()
labeled_detections = preprocessor_ztf.preprocess(labeled_detections)

# Save data
labeled_detections.to_pickle('dataset_detections.pkl')
labeled_non_detections.to_pickle('dataset_non_detections.pkl')
labels.to_pickle('dataset_labels.pkl')
