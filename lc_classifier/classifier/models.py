import numpy as np
import pandas as pd
import tensorflow as tf
import os
from pathlib import Path
import pickle
import wget
import tarfile
from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from lc_classifier.classifier.preprocessing import FeaturePreprocessor
from lc_classifier.classifier.preprocessing import MLPFeaturePreprocessor
from lc_classifier.classifier.preprocessing import intersect_oids_in_dataframes
from lc_classifier.classifier.mlp import MLP
from abc import ABC, abstractmethod
from typing import List


MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
PICKLE_PATH = os.path.join(MODEL_PATH, "pickles")


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    def predict(self, samples: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(samples)
        predicted_class = probs.idxmax(axis=1)
        predicted_class_df = pd.DataFrame(
            predicted_class, columns=["classALeRCE"], index=samples.index
        )
        predicted_class_df.index.name = samples.index.name
        return predicted_class_df

    # compatibility method for new lc_classification step
    def can_predict(self, data):
        return True

    @abstractmethod
    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_list_of_classes(self) -> list:
        pass

    @abstractmethod
    def save_model(self, directory: str) -> None:
        pass

    @abstractmethod
    def load_model(self, directory: str) -> None:
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
            max_features="sqrt",
            max_depth=None,
            n_jobs=1,
            class_weight=None,
            criterion="entropy",
            min_samples_split=2,
            min_samples_leaf=1,
        )
        self.feature_preprocessor = FeaturePreprocessor()
        self.feature_list = None
        self.model_filename = "baseline_rf.pkl"

    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame):
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples = self.feature_preprocessor.remove_duplicates(samples)

        # intersect samples and labels
        samples, labels = intersect_oids_in_dataframes(samples, labels)

        self.feature_list = samples.columns
        samples_np_array = samples.values
        labels_np_array = labels["classALeRCE"].loc[samples.index].values
        self.random_forest_classifier.fit(samples_np_array, labels_np_array)

    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples_np_array = samples[self.feature_list].values
        predicted_probs = self.random_forest_classifier.predict_proba(samples_np_array)
        predicted_probs_df = pd.DataFrame(
            predicted_probs,
            columns=self.get_list_of_classes(),
            index=samples.index.values,
        )
        predicted_probs_df.index.name = "oid"
        return predicted_probs_df

    def get_list_of_classes(self) -> list:
        return self.random_forest_classifier.classes_

    def save_model(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(directory, self.model_filename), "wb") as f:
            pickle.dump(self.random_forest_classifier, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(directory, "feature_list.pkl"), "wb") as f:
            pickle.dump(self.feature_list, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, directory: str) -> None:
        with open(os.path.join(directory, self.model_filename), 'rb') as f:
            rf = pickle.load(f)

        self.random_forest_classifier = rf

        with open(os.path.join(directory, "feature_list.pkl"), 'rb') as f:
            self.feature_list = pickle.load(f)


class HierarchicalRandomForest(BaseClassifier):
    MODEL_NAME = "hierarchical_random_forest"
    MODEL_VERSION = "1.1.1"
    MODEL_VERSION_NAME = f"{MODEL_NAME}_{MODEL_VERSION}"
    MODEL_PICKLE_PATH = os.path.join(PICKLE_PATH, f"{MODEL_VERSION_NAME}")
    taxonomy_dictionary = {
        'Stochastic': ['AGN', 'Blazar', 'CV/Nova', 'QSO', 'YSO'],
        'Periodic': ['CEP', 'DSCT', 'E', 'LPV', 'Periodic-Other', 'RRL'],
        'Transient': ['SLSN', 'SNII', 'SNIa', 'SNIbc']
    }

    def __init__(self,
                 taxonomy_dictionary=None,
                 non_used_features=None,
                 n_trees=500,
                 n_jobs=1,
                 verbose: bool = False):

        self.verbose = verbose
        if self.verbose:
            verbose_number = 11
        else:
            verbose_number = 0
            
        self.top_classifier = RandomForestClassifier(
            n_estimators=n_trees, max_depth=None,
            max_features="sqrt",
            n_jobs=n_jobs,
            verbose=verbose_number
        )

        self.stochastic_classifier = RandomForestClassifier(
            n_estimators=n_trees, max_depth=None,
            max_features=0.2,
            n_jobs=n_jobs,
            verbose=verbose_number
        )

        self.periodic_classifier = RandomForestClassifier(
            n_estimators=n_trees, max_depth=None,
            max_features="sqrt",
            n_jobs=n_jobs,
            verbose=verbose_number
        )

        self.transient_classifier = RandomForestClassifier(
            n_estimators=n_trees, max_depth=None,
            max_features="sqrt",
            n_jobs=n_jobs,
            verbose=verbose_number
        )

        self.feature_preprocessor = FeaturePreprocessor(
            non_used_features=non_used_features
        )

        self.feature_list = None

        if taxonomy_dictionary is not None:
            self.taxonomy_dictionary = taxonomy_dictionary

        self.inverted_dictionary = invert_dictionary(self.taxonomy_dictionary)
        self.pickles = {
            "features_list": "features_RF_model.pkl",
            "top_rf": "top_level_BRF_model.pkl",
            "periodic_rf": "periodic_level_BRF_model.pkl",
            "stochastic_rf": "stochastic_level_BRF_model.pkl",
            "transient_rf": "transient_level_BRF_model.pkl",
        }
        self.url_model = f"https://assets.alerce.online/pipeline/hierarchical_rf_{self.MODEL_VERSION}/"

    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame) -> None:
        labels = labels.copy()

        # Check that the received labels are in the taxonomy
        feeded_labels = labels.classALeRCE.unique()
        expected_labels = self.inverted_dictionary.keys()

        for label in feeded_labels:
            if label not in expected_labels:
                raise Exception(f"{label} is not in the taxonomy dictionary")

        # Create top class
        labels["top_class"] = labels["classALeRCE"].map(self.inverted_dictionary)

        # Preprocessing
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples = self.feature_preprocessor.remove_duplicates(samples)
        samples, labels = intersect_oids_in_dataframes(samples, labels)

        # Save list of features to know their order
        self.feature_list = samples.columns

        # Train top classifier
        if self.verbose:
            print("training top classifier")
        self.top_classifier.fit(samples.values, labels["top_class"].values)

        # Train specialized classifiers
        if self.verbose:
            print("training stochastic classifier")
        is_stochastic = labels["top_class"] == "Stochastic"
        self.stochastic_classifier.fit(
            samples[is_stochastic].values, labels[is_stochastic]["classALeRCE"].values
        )

        if self.verbose:
            print("training periodic classifier")
        is_periodic = labels["top_class"] == "Periodic"
        self.periodic_classifier.fit(
            samples[is_periodic].values, labels[is_periodic]["classALeRCE"].values
        )

        if self.verbose:
            print("training transient classifier")
        is_transient = labels["top_class"] == "Transient"
        self.transient_classifier.fit(
            samples[is_transient].values, labels[is_transient]["classALeRCE"].values
        )

    def check_missing_features(self, columns, feature_list):
        missing = set(feature_list).difference(set(columns))
        return missing

    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        missing = self.check_missing_features(samples.columns, self.feature_list)
        if len(missing) > 0:
            raise Exception(f"Missing features: {missing}")

        samples = samples[self.feature_list]
        samples = self.feature_preprocessor.preprocess_features(samples)

        top_probs = self.top_classifier.predict_proba(samples.values)

        stochastic_probs = self.stochastic_classifier.predict_proba(samples.values)
        periodic_probs = self.periodic_classifier.predict_proba(samples.values)
        transient_probs = self.transient_classifier.predict_proba(samples.values)

        stochastic_index = self.top_classifier.classes_.tolist().index("Stochastic")
        periodic_index = self.top_classifier.classes_.tolist().index("Periodic")
        transient_index = self.top_classifier.classes_.tolist().index("Transient")

        stochastic_probs = stochastic_probs * top_probs[:, stochastic_index].reshape(
            [-1, 1]
        )
        periodic_probs = periodic_probs * top_probs[:, periodic_index].reshape([-1, 1])
        transient_probs = transient_probs * top_probs[:, transient_index].reshape(
            [-1, 1]
        )

        # This line must have the same order as in get_list_of_classes()
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
            + self.transient_classifier.classes_.tolist()
        )
        return final_columns

    def save_model(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(directory, self.pickles["top_rf"]), "wb") as f:
            pickle.dump(self.top_classifier, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, self.pickles["stochastic_rf"]), "wb") as f:
            pickle.dump(self.stochastic_classifier, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, self.pickles["periodic_rf"]), "wb") as f:
            pickle.dump(self.periodic_classifier, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, self.pickles["transient_rf"]), "wb") as f:
            pickle.dump(self.transient_classifier, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, self.pickles["features_list"]), "wb") as f:
            pickle.dump(self.feature_list, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, directory: str) -> None:
        with open(os.path.join(directory, self.pickles["top_rf"]), 'rb') as f:
            self.top_classifier = pickle.load(f)

        with open(os.path.join(directory, self.pickles["stochastic_rf"]), 'rb') as f:
            self.stochastic_classifier = pickle.load(f)

        with open(os.path.join(directory, self.pickles["periodic_rf"]), 'rb') as f:
            self.periodic_classifier = pickle.load(f)

        with open(os.path.join(directory, self.pickles["transient_rf"]), 'rb') as f:
            self.transient_classifier = pickle.load(f)

        with open(os.path.join(directory, self.pickles["features_list"]), 'rb') as f:
            self.feature_list = pickle.load(f)

        self.check_loaded_models()

    def check_loaded_models(self):
        assert set(self.top_classifier.classes_.tolist()) \
               == set(self.taxonomy_dictionary.keys())

        assert set(self.stochastic_classifier.classes_.tolist()) \
               == set(self.taxonomy_dictionary["Stochastic"])

        assert set(self.transient_classifier.classes_.tolist()) \
               == set(self.taxonomy_dictionary["Transient"])

        assert set(self.periodic_classifier.classes_.tolist()) \
               == set(self.taxonomy_dictionary["Periodic"])

    def download_model(self):
        print(self.MODEL_PICKLE_PATH)
        print(self.url_model)
        if not os.path.exists(self.MODEL_PICKLE_PATH):
            os.makedirs(self.MODEL_PICKLE_PATH)

        compressed_extension = '.tar.gz'
        if self.url_model[-len(compressed_extension):] == compressed_extension:
            filename = 'model' + compressed_extension
            full_filename = os.path.join(self.MODEL_PICKLE_PATH, filename)
            wget.download(
                self.url_model,
                full_filename)
            file = tarfile.open(full_filename)
            file.extractall(self.MODEL_PICKLE_PATH)
            file.close()
        else:
            for pkl in self.pickles.values():
                tmp_path = os.path.join(self.MODEL_PICKLE_PATH, pkl)
                if not os.path.exists(tmp_path):
                    wget.download(os.path.join(self.url_model, pkl), tmp_path)

    def predict_in_pipeline(self, input_features: pd.DataFrame) -> dict:
        if not isinstance(input_features, pd.core.frame.DataFrame):
            raise TypeError("predict_in_pipeline expects a DataFrame.")

        missing = self.check_missing_features(input_features.columns, self.feature_list)
        if len(missing) > 0:
            raise Exception(f"Missing features: {missing}")

        input_features = input_features[self.feature_list]

        input_features = self.feature_preprocessor.preprocess_features(input_features)
        prob_root = pd.DataFrame(
            self.top_classifier.predict_proba(input_features),
            columns=self.top_classifier.classes_,
            index=input_features.index,
        )

        prob_children = []
        resp_children = {}
        child_models = [
            self.periodic_classifier,
            self.stochastic_classifier,
            self.transient_classifier,
        ]
        child_names = ["Periodic", "Stochastic", "Transient"]

        for name, model in zip(child_names, child_models):
            prob_child = pd.DataFrame(
                model.predict_proba(input_features),
                columns=model.classes_,
                index=input_features.index,
            )

            resp_children[name] = prob_child
            prob_child = prob_child.mul(prob_root[name].values, axis=0)
            prob_children.append(prob_child)
        prob_all = pd.concat(prob_children, axis=1, sort=False)

        return {
            "hierarchical": {"top": prob_root, "children": resp_children},
            "probabilities": prob_all,
            "class": prob_all.idxmax(axis=1),
        }
    
    def can_predict(self, data):
        """
        data is an inputDTO.
        We will use data.feature to check for missing features
        We will use data.detections to check if there are any ztf detections
        """
        missing = self.check_missing_features(data.features.columns, self.feature_list)
        features_ok = len(missing) == 0
        
        try:
            detections_ok = (data.detections["tid"] == "ztf").any()
        except Exception:
            detections_ok = False

        return features_ok and detections_ok


class ElasticcRandomForest(HierarchicalRandomForest):
    MODEL_NAME = "elasticc_BHRF"
    MODEL_VERSION = "2.0.0"
    MODEL_VERSION_NAME = f"{MODEL_NAME}_{MODEL_VERSION}"
    MODEL_PICKLE_PATH = os.path.join(PICKLE_PATH, f"{MODEL_VERSION_NAME}")

    def __init__(self,
        taxonomy_dictionary,
        feature_list_dict: dict,
        sampling_strategy=None,
        n_trees=500,
        n_jobs=1,
        verbose: bool = False,
        model_name: str = None,
        ):

        self.verbose = verbose
        if self.verbose:
            verbose_number = 11
        else:
            verbose_number = 0
        
        if model_name is not None:
            self.MODEL_NAME = model_name
            self.MODEL_VERSION_NAME = f"{model_name}_{self.MODEL_VERSION}"
            self.MODEL_PICKLE_PATH = os.path.join(PICKLE_PATH, f"{self.MODEL_VERSION_NAME}")

        max_depth = None
        max_features = "sqrt"
        min_samples_leaf = 1
        min_samples_split = 2

        if sampling_strategy is None:
            sampling_strategy = {
                'Top': 'auto',
                'Stochastic': 'auto',
                'Periodic': 'auto',
                'Transient': 'auto'
            }

        # imblearn uses a weight mask, not slicing
        max_samples = None  # 10000
        bootstrap = False
            
        self.top_classifier = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            sampling_strategy=sampling_strategy['Top'],
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            verbose=verbose_number
        )

        self.stochastic_classifier = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            sampling_strategy=sampling_strategy['Stochastic'],
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            verbose=verbose_number
        )

        self.periodic_classifier = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            sampling_strategy=sampling_strategy['Periodic'],
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            verbose=verbose_number
        )

        self.transient_classifier = RandomForestClassifier(
            n_estimators=350,  # n_trees,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=0.00003,
            sampling_strategy=sampling_strategy['Transient'],
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            verbose=verbose_number
        )
        
        self.feature_preprocessor = FeaturePreprocessor()

        # each tree has its own features
        self.feature_list_dict = feature_list_dict

        self.taxonomy_dictionary = taxonomy_dictionary

        self.inverted_dictionary = invert_dictionary(self.taxonomy_dictionary)
        self.pickles = {
            "feature_list_dict": "features_HRF_model.pkl",
            "top_rf": "top_level_HRF_model.pkl",
            "periodic_rf": "periodic_level_HRF_model.pkl",
            "stochastic_rf": "stochastic_level_HRF_model.pkl",
            "transient_rf": "transient_level_HRF_model.pkl",
        }

    def check_missing_features(self, available_features):
        available_features = set(available_features)
        required_features = set(np.concatenate(
            list(self.feature_list_dict.values())))

        missing_features = required_features - available_features
        if len(missing_features) != 0:
            raise ValueError(f'missing required features: {missing_features}')

    def fit(self, samples: pd.DataFrame, labels: pd.DataFrame) -> None:
        labels = labels.copy()

        # Check that the received labels are in the taxonomy
        feeded_labels = labels.classALeRCE.unique()
        expected_labels = self.inverted_dictionary.keys()

        for label in feeded_labels:
            if label not in expected_labels:
                raise Exception(f"{label} is not in the taxonomy dictionary")

        # Create top class
        labels["top_class"] = labels["classALeRCE"].map(self.inverted_dictionary)

        # Preprocessing
        samples = self.feature_preprocessor.preprocess_features(samples)
        samples = self.feature_preprocessor.remove_duplicates(samples)
        samples, labels = intersect_oids_in_dataframes(samples, labels)

        self.check_missing_features(samples.columns)
        
        # Train top classifier
        if self.verbose:
            print("training top classifier")

        rus = RandomUnderSampler()
        selected_top_snids, _ = rus.fit_resample(
            samples.index.values.reshape(-1, 1),
            labels['classALeRCE'])
        selected_top_snids = selected_top_snids.flatten()
        
        self.top_classifier.fit(
            samples.loc[selected_top_snids][self.feature_list_dict['top']].values,
            labels.loc[selected_top_snids]["top_class"].values)

        # Train specialized classifiers
        if self.verbose:
            print("training stochastic classifier")
        is_stochastic = labels["top_class"] == "Stochastic"
        self.stochastic_classifier.fit(
            samples[is_stochastic][self.feature_list_dict['stochastic']].values,
            labels[is_stochastic]["classALeRCE"].values)

        if self.verbose:
            print("training periodic classifier")
        is_periodic = labels["top_class"] == "Periodic"
        self.periodic_classifier.fit(
            samples[is_periodic][self.feature_list_dict['periodic']].values,
            labels[is_periodic]["classALeRCE"].values)

        if self.verbose:
            print("training transient classifier")
        is_transient = labels["top_class"] == "Transient"
        self.transient_classifier.fit(
            samples[is_transient][self.feature_list_dict['transient']].values,
            labels[is_transient]["classALeRCE"].values)

    def save_model(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(directory, self.pickles["top_rf"]), "wb") as f:
            pickle.dump(self.top_classifier, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, self.pickles["stochastic_rf"]), "wb") as f:
            pickle.dump(self.stochastic_classifier, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, self.pickles["periodic_rf"]), "wb") as f:
            pickle.dump(self.periodic_classifier, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, self.pickles["transient_rf"]), "wb") as f:
            pickle.dump(self.transient_classifier, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, self.pickles["feature_list_dict"]), "wb") as f:
            pickle.dump(self.feature_list_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, directory: str, n_jobs: int) -> None:
        with open(os.path.join(directory, self.pickles["top_rf"]), 'rb') as f:
            self.top_classifier = pickle.load(f)
            self.top_classifier.n_jobs = n_jobs

        with open(os.path.join(directory, self.pickles["stochastic_rf"]), 'rb') as f:
            self.stochastic_classifier = pickle.load(f)
            self.stochastic_classifier.n_jobs = n_jobs
            
        with open(os.path.join(directory, self.pickles["periodic_rf"]), 'rb') as f:
            self.periodic_classifier = pickle.load(f)
            self.periodic_classifier.n_jobs = n_jobs

        with open(os.path.join(directory, self.pickles["transient_rf"]), 'rb') as f:
            self.transient_classifier = pickle.load(f)
            self.transient_classifier.n_jobs = n_jobs

        with open(os.path.join(directory, self.pickles["feature_list_dict"]), 'rb') as f:
            self.feature_list_dict = pickle.load(f)

        self.check_loaded_models()

    def check_loaded_models(self):
        assert set(self.top_classifier.classes_.tolist()) \
               == set(self.taxonomy_dictionary.keys())

        assert set(self.stochastic_classifier.classes_.tolist()) \
               == set(self.taxonomy_dictionary["Stochastic"])

        assert set(self.transient_classifier.classes_.tolist()) \
               == set(self.taxonomy_dictionary["Transient"])

        assert set(self.periodic_classifier.classes_.tolist()) \
               == set(self.taxonomy_dictionary["Periodic"])

    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        self.check_missing_features(samples.columns)
        
        samples = self.feature_preprocessor.preprocess_features(samples)

        top_probs = self.top_classifier.predict_proba(
            samples[self.feature_list_dict['top']].values)

        stochastic_probs = self.stochastic_classifier.predict_proba(
            samples[self.feature_list_dict['stochastic']].values)
        periodic_probs = self.periodic_classifier.predict_proba(
            samples[self.feature_list_dict['periodic']].values)
        transient_probs = self.transient_classifier.predict_proba(
            samples[self.feature_list_dict['transient']].values)

        stochastic_index = self.top_classifier.classes_.tolist().index("Stochastic")
        periodic_index = self.top_classifier.classes_.tolist().index("Periodic")
        transient_index = self.top_classifier.classes_.tolist().index("Transient")

        stochastic_probs = stochastic_probs * top_probs[:, stochastic_index].reshape(
            [-1, 1]
        )
        periodic_probs = periodic_probs * top_probs[:, periodic_index].reshape([-1, 1])
        transient_probs = transient_probs * top_probs[:, transient_index].reshape(
            [-1, 1]
        )

        # This line must have the same order as in get_list_of_classes()
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


class ElasticcMLP(BaseClassifier):
    def __init__(
            self, 
            list_of_classes: List[str], 
            non_used_features: List[str] = None,
            n_features: int = 498
            ):
        self.feature_preprocessor = MLPFeaturePreprocessor(
            non_used_features=non_used_features
        )
        self.list_of_classes = list_of_classes
        self.mlp = MLP(self.list_of_classes)
        self.batch_size = 128
        self.n_features = n_features

    def fit(self,
            x_training: pd.DataFrame,
            y_training: pd.DataFrame,
            y_teacher: pd.DataFrame,
            x_validation: pd.DataFrame,
            y_validation: pd.DataFrame,
            x_test: pd.DataFrame,
            y_test: pd.DataFrame) -> None:

        print('shuffling training set')
        new_order = np.random.permutation(len(x_training))
        x_training = x_training.iloc[new_order]
        y_training = y_training.iloc[new_order]
        y_teacher = y_teacher.iloc[new_order]

        print('tf dataset training')
        training_dataset = self._tf_dataset_from_dataframes(
            x_training, y_training, y_teacher, 
            balanced=True, fit_preprocessor=True)
        print('tf dataset validation')
        validation_dataset = self._tf_dataset_from_dataframes(
            x_validation, y_validation, None, 
            balanced=False, fit_preprocessor=False)
        print('tf dataset test')
        test_dataset = self._tf_dataset_from_dataframes(
            x_test, y_test, None, 
            balanced=False, fit_preprocessor=False)

        training_dataset = training_dataset.shuffle(self.batch_size*25).batch(self.batch_size)
        validation_dataset = validation_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        
        print('training')
        self.mlp.train(training_dataset, validation_dataset, test_dataset)

    def _tf_dataset_from_dataframes(
            self, 
            x_dataframe: pd.DataFrame, 
            y_dataframe: pd.DataFrame, 
            y_teacher_df: pd.DataFrame,
            balanced: bool, 
            fit_preprocessor: bool = False):
        
        if fit_preprocessor:
            self.feature_preprocessor.fit(x_dataframe)
        
        samples = self.feature_preprocessor.preprocess_features(
            x_dataframe).values.astype(np.float32)
        labels = self._label_list_to_one_hot(y_dataframe['classALeRCE'].values)
        if y_teacher_df is not None:
            y_teacher_np = y_teacher_df[self.list_of_classes].values

            if balanced:
                datasets_from_class = []
                for class_i in range(len(self.list_of_classes)):
                    from_class = labels[:, class_i].astype(bool)
                    samples_from_class = samples[from_class]
                    labels_from_class = labels[from_class]
                    y_teacher_from_class = y_teacher_np[from_class]

                    tf_dataset_from_class = tf.data.Dataset.from_tensor_slices(
                        (samples_from_class, labels_from_class, y_teacher_from_class))
                    tf_dataset_from_class = tf_dataset_from_class\
                        .repeat().shuffle(100, reshuffle_each_iteration=True).prefetch(20)
                    datasets_from_class.append(tf_dataset_from_class)
                tf_dataset = tf.data.Dataset.sample_from_datasets(
                    datasets_from_class)
            else:
                tf_dataset = tf.data.Dataset.from_tensor_slices(
                    (samples, labels, y_teacher_np))
        else:
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (samples, labels))
        return tf_dataset
        
    def _label_list_to_one_hot(self, label_list: np.ndarray):
        matches = np.stack(
            [label_list == s for s in self.list_of_classes], axis=-1)
        onehot_labels = matches.astype(np.float32)
        return onehot_labels
        
    def predict_proba(self, samples: pd.DataFrame) -> pd.DataFrame:
        # feature preprocessor guarantees output columns in the right order
        samples_index = samples.index.values
        samples = self.feature_preprocessor.preprocess_features(
            samples).values.astype(np.float32)

        n_batches = int(np.ceil(len(samples) / self.batch_size))
        probs = []
        for i_batch in range(n_batches):
            batch_samples = samples[i_batch*self.batch_size:(i_batch+1)*self.batch_size]
            batch_probs = self.mlp(batch_samples, training=False, logits=False)
            probs.append(batch_probs)

        probs = np.concatenate(probs, axis=0)
        probs_df = pd.DataFrame(
            data=probs,
            columns=self.list_of_classes,
            index=samples_index
        )
        return probs_df

    def get_list_of_classes(self) -> List[str]:
        return self.list_of_classes

    def save_model(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.mkdir(directory)

        self.mlp.save_weights(
            os.path.join(directory, 'mlp.h5')
        )

        self.feature_preprocessor.save(
            os.path.join(directory, 'feature_preprocessor.pkl')
        )

    def load_model(self, directory: str) -> None:
        self.mlp(np.random.randn(self.batch_size, self.n_features))
        self.mlp.load_weights(
            os.path.join(directory, 'mlp.h5')
        )
        self.feature_preprocessor.load(
            os.path.join(directory, 'feature_preprocessor.pkl')
        )
