import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier
from late_classifier.classifier.preprocessing import FeaturePreprocessor
from late_classifier.classifier.preprocessing import intersect_oids_in_dataframes
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, samples: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_list_of_classes(self) -> list:
        pass


def invert_dictionary(dictionary):
    inverted_dictionary = {}
    for top_group, list_of_classes in dictionary.items():
        for astro_class in list_of_classes:
            inverted_dictionary[astro_class] = top_group
    return inverted_dictionary


class BaselineRandomForest(BaseClassifier):
    def __init__(self):
        self.random_forest_classifier = RandomForestClassifier(
            n_estimators=500,
            max_features='auto',
            max_depth=None,
            n_jobs=1,
            class_weight=None,
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1)
        self.feature_preprocessor = FeaturePreprocessor()
        self.feature_list = None

    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame):
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples = self.feature_preprocessor.remove_duplicates(samples)

        # intersect samples and labels
        samples, labels = intersect_oids_in_dataframes(samples, labels)

        self.feature_list = samples.columns
        samples_np_array = samples.values
        labels_np_array = labels['classALeRCE'].loc[samples.index].values
        self.random_forest_classifier.fit(samples_np_array, labels_np_array)

    def predict(self, samples: pd.DataFrame) -> pd.DataFrame:
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples_np_array = samples[self.feature_list].values
        predictions = self.random_forest_classifier.predict(samples_np_array)
        predictions_df = pd.DataFrame(
            predictions,
            columns=['classALeRCE'],
            index=samples.index.values
        )
        predictions_df.index.name = 'oid'
        return predictions_df

    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples_np_array = samples[self.feature_list].values
        predicted_probs = self.random_forest_classifier.predict_proba(samples_np_array)
        predicted_probs_df = pd.DataFrame(
            predicted_probs,
            columns=self.get_list_of_classes(),
            index=samples.index.values
        )
        predicted_probs_df.index.name = 'oid'
        return predicted_probs_df

    def get_list_of_classes(self) -> list:
        return self.random_forest_classifier.classes_


class HierarchicalRandomForest(BaseClassifier):
    def __init__(self, taxonomy_dictionary):
        n_trees = 500
        self.top_classifier = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=None,
            max_features='auto'
        )

        self.stochastic_classifier = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=None,
            max_features=0.2
        )

        self.periodic_classifier = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=None,
            max_features='auto'
        )

        self.transient_classifier = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=None,
            max_features='auto'
        )

        self.feature_preprocessor = FeaturePreprocessor()

        self.taxonomy_dictionary = taxonomy_dictionary
        self.inverted_dictionary = invert_dictionary(self.taxonomy_dictionary)

    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame) -> None:
        labels = labels.copy()

        # Check that the received labels are in the taxonomy
        feeded_labels = labels.classALeRCE.unique()
        expected_labels = self.inverted_dictionary.keys()

        for label in feeded_labels:
            if label not in expected_labels:
                raise Exception(f'{label} is not in the taxonomy dictionary')

        # Create top class
        labels['top_class'] = labels['classALeRCE'].map(self.inverted_dictionary)

        # Preprocessing
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples = self.feature_preprocessor.remove_duplicates(samples)
        samples, labels = intersect_oids_in_dataframes(samples, labels)

        # Train top classifier
        self.top_classifier.fit(samples.values, labels['top_class'].values)

        # Train specialized classifiers
        is_stochastic = labels['top_class'] == 'stochastic'
        self.stochastic_classifier.fit(
            samples[is_stochastic].values,
            labels[is_stochastic]['classALeRCE'].values
        )

        is_periodic = labels['top_class'] == 'periodic'
        self.periodic_classifier.fit(
            samples[is_periodic].values,
            labels[is_periodic]['classALeRCE'].values
        )

        is_transient = labels['top_class'] == 'transient'
        self.transient_classifier.fit(
            samples[is_transient].values,
            labels[is_transient]['classALeRCE'].values
        )

        print(self.top_classifier.classes_)

    def predict(self, samples: pd.DataFrame) -> pd.DataFrame:
        pass

    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        samples = self.feature_preprocessor.preprocess_features(samples)

        top_probs = self.top_classifier.predict_proba(samples.values)

        stochastic_probs = self.stochastic_classifier.predict_proba(samples.values)
        periodic_probs = self.periodic_classifier.predict_proba(samples.values)
        transient_probs = self.transient_classifier.predict_proba(samples.values)

        stochastic_index = self.top_classifier.classes_.tolist().index('stochastic')
        periodic_index = self.top_classifier.classes_.tolist().index('periodic')
        transient_index = self.top_classifier.classes_.tolist().index('transient')

        stochastic_probs = stochastic_probs * top_probs[:, stochastic_index].reshape([-1, 1])
        periodic_probs = periodic_probs * top_probs[:, periodic_index].reshape([-1, 1])
        transient_probs = transient_probs * top_probs[:, transient_index].reshape([-1, 1])

        final_probs = np.concatenate(
            [stochastic_probs, periodic_probs, transient_probs],
            axis=1
        )

        df = pd.DataFrame(
            data=final_probs,
            index=samples.index,
            columns=self.get_list_of_classes()
        )
        df.index.name = samples.index.name
        return df

    def get_list_of_classes(self) -> list:
        final_columns = (
                self.stochastic_classifier.classes_.tolist()
                + self.periodic_classifier.classes_.tolist()
                + self.transient_classifier.classes_.tolist())
        return final_columns
