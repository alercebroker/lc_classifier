import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier
from late_classifier.classifier.preprocessing import FeaturePreprocessor
from late_classifier.classifier.preprocessing import intersect_oids_in_dataframes


class BaselineRandomForest:
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
