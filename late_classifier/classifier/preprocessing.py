import pandas as pd
import numpy as np
import json


class FeaturePreprocessor:
    def __init__(self, non_used_features=None):
        self.non_used_features = non_used_features

    def preprocess_features(self, features) -> pd.DataFrame:
        if self.non_used_features is not None:
            new_columns = [
                feature for feature in features.columns
                if feature not in self.non_used_features]
            features = features[new_columns]
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(-999.0)
        features[features > 1e32] = 0.0
        return features

    def remove_duplicates(self, features):
        features = features.loc[~features.index.duplicated(keep='first')]
        return features


class LabelPreprocessor:
    """Groups and filters classes"""
    def __init__(self, class_dictionary_filename):
        with open(class_dictionary_filename) as f:
            self.class_dictionary = json.load(f)['class_dictionary']
        self.used_classes = self.class_dictionary.keys()

    def preprocess_labels(self, labels):
        labels = labels[labels['classALeRCE'].isin(self.used_classes)]
        labels = labels.copy()
        labels['classALeRCE'] = labels['classALeRCE'].replace(self.class_dictionary)
        return labels


def intersect_oids_in_dataframes(df1, df2):
    oid_df1 = set(df1.index.values.tolist())
    oid_df2 = set(df2.index.values.tolist())
    oid_intersection = oid_df1.intersection(oid_df2)
    oid_intersection = sorted(list(oid_intersection))
    new_df1 = df1.loc[oid_intersection]
    new_df2 = df2.loc[oid_intersection]
    assert len(new_df1) == len(new_df2)
    return new_df1, new_df2
