import unittest
from unittest import mock
import pandas as pd
import random

from lc_classifier.classifier.models import HierarchicalRandomForest

class MockDTO:
    def __init__(self, detections, features):
        self._detections = detections
        self._featrures = features
    
    @property
    def detections(self):
        return self._detections
    
    @property
    def features(self):
        return self._featrures
    
class TestHierarchicalRF(unittest.TestCase):
    def setUp(self) -> None:
        self.train_features = pd.read_csv('data_examples/2000_features.csv')
        self.train_labels = pd.read_csv('data_examples/2000_labels.csv')
        del self.train_features["objectId"]

        self.test_features = pd.read_csv('data_examples/100_objects_features.csv')
        self.test_labels = pd.read_csv('data_examples/100_objects_labels.csv')
        del self.test_labels["objectId"]

        self.model_trained = HierarchicalRandomForest()
        self.model_trained.download_model()
        self.model_trained.load_model(self.model_trained.MODEL_PICKLE_PATH)

        self.taxonomy = {
            'Stochastic': ['LPV', 'QSO', 'YSO', 'CV/Nova', 'Blazar', 'AGN'],
            'Periodic': ['RRL', 'E', 'DSCT', 'CEP', 'Periodic-Other'],
            'Transient': ['SNIa', 'SNIbc', 'SLSN']
        }
        self.model_silly = HierarchicalRandomForest(
            taxonomy_dictionary=self.taxonomy)

    def test_predict(self):
        predicted_classes = self.model_trained.predict(self.test_features)
        self.assertListEqual(
            list(predicted_classes["classALeRCE"][:5]),
            list(self.test_labels["classALeRCE"][:5])
        )

    def test_predict_in_pipeline(self):
        prediction = self.model_trained.predict_in_pipeline(
            self.test_features.iloc[[0]])
        self.assertIsInstance(prediction, dict)
        self.assertEqual(prediction["class"].iloc[0], "E")

    def test_fit(self):
        self.model_silly.fit(self.train_features, self.train_labels)
        probabilities = self.model_silly.predict_proba(self.train_features)
        predicted_classes = self.model_silly.predict(self.test_features)
        self.assertEqual(len(probabilities), 2000)
        self.assertEqual(len(predicted_classes), 98)

    def test_io(self):
        self.model_silly.fit(self.train_features, self.train_labels)
        self.model_silly.save_model("")
        self.model_silly.load_model("")

class TestHierarchicalRFCanPredict(unittest.TestCase):
    def setUp(self) -> None:
        self.test_features = pd.read_csv('data_examples/100_objects_features.csv')

        self.model = HierarchicalRandomForest()
        self.model.download_model()
        self.model.load_model(self.model.MODEL_PICKLE_PATH)

    def test_can_predict_true(self):
        test_detections = pd.DataFrame(
            {
                "candid": str(random.randint(1000000, 9000000)),
                "oid": "ZTFoid",
                "aid": "aid",
                "tid": "ztf",
                "mjd": random.uniform(59000, 60000),
                "sid": "ZTF",
                "fid": "g",
                "pid": random.randint(1000000, 9000000),
                "ra": random.uniform(-90, 90),
                "e_ra": random.uniform(-90, 90),
                "dec": random.uniform(-90, 90),
                "e_dec": random.uniform(-90, 90),
                "mag": random.uniform(15, 20),
                "e_mag": random.uniform(0, 1),
                "mag_corr": random.uniform(15, 20),
                "e_mag_corr": random.uniform(0, 1),
                "e_mag_corr_ext": random.uniform(0, 1),
                "isdiffpos": random.choice([-1, 1]),
                "corrected": random.choice([True, False]),
                "dubious": random.choice([True, False]),
                "has_stamp": random.choice([True, False]),
                "stellar": random.choice([True, False]),
                "new": random.choice([True, False]),
                "forced": random.choice([True, False]),
                "extra_fields": {},
            },{
                "candid": str(random.randint(1000000, 9000000)),
                "oid": "ZTFoid2",
                "aid": "aid2",
                "tid": "ztf",
                "mjd": random.uniform(59000, 60000),
                "sid": "ZTF",
                "fid": "g",
                "pid": random.randint(1000000, 9000000),
                "ra": random.uniform(-90, 90),
                "e_ra": random.uniform(-90, 90),
                "dec": random.uniform(-90, 90),
                "e_dec": random.uniform(-90, 90),
                "mag": random.uniform(15, 20),
                "e_mag": random.uniform(0, 1),
                "mag_corr": random.uniform(15, 20),
                "e_mag_corr": random.uniform(0, 1),
                "e_mag_corr_ext": random.uniform(0, 1),
                "isdiffpos": random.choice([-1, 1]),
                "corrected": random.choice([True, False]),
                "dubious": random.choice([True, False]),
                "has_stamp": random.choice([True, False]),
                "stellar": random.choice([True, False]),
                "new": random.choice([True, False]),
                "forced": random.choice([True, False]),
                "extra_fields": {},
            }
        )
        data_for_test = MockDTO(test_detections, self.test_features)
            
        can_predict, _ = self.model.can_predict(data_for_test)

        self.assertTrue(can_predict)

    def test_can_precit_false(self):
        # No features With alerts
        test_detections = pd.DataFrame(
            {
                "candid": str(random.randint(1000000, 9000000)),
                "oid": "ZTFoid",
                "aid": "aid",
                "tid": "ztf",
                "mjd": random.uniform(59000, 60000),
                "sid": "ZTF",
                "fid": "g",
                "pid": random.randint(1000000, 9000000),
                "ra": random.uniform(-90, 90),
                "e_ra": random.uniform(-90, 90),
                "dec": random.uniform(-90, 90),
                "e_dec": random.uniform(-90, 90),
                "mag": random.uniform(15, 20),
                "e_mag": random.uniform(0, 1),
                "mag_corr": random.uniform(15, 20),
                "e_mag_corr": random.uniform(0, 1),
                "e_mag_corr_ext": random.uniform(0, 1),
                "isdiffpos": random.choice([-1, 1]),
                "corrected": random.choice([True, False]),
                "dubious": random.choice([True, False]),
                "has_stamp": random.choice([True, False]),
                "stellar": random.choice([True, False]),
                "new": random.choice([True, False]),
                "forced": random.choice([True, False]),
                "extra_fields": {},
            },{
                "candid": str(random.randint(1000000, 9000000)),
                "oid": "ZTFoid2",
                "aid": "aid2",
                "tid": "ztf",
                "mjd": random.uniform(59000, 60000),
                "sid": "ZTF",
                "fid": "g",
                "pid": random.randint(1000000, 9000000),
                "ra": random.uniform(-90, 90),
                "e_ra": random.uniform(-90, 90),
                "dec": random.uniform(-90, 90),
                "e_dec": random.uniform(-90, 90),
                "mag": random.uniform(15, 20),
                "e_mag": random.uniform(0, 1),
                "mag_corr": random.uniform(15, 20),
                "e_mag_corr": random.uniform(0, 1),
                "e_mag_corr_ext": random.uniform(0, 1),
                "isdiffpos": random.choice([-1, 1]),
                "corrected": random.choice([True, False]),
                "dubious": random.choice([True, False]),
                "has_stamp": random.choice([True, False]),
                "stellar": random.choice([True, False]),
                "new": random.choice([True, False]),
                "forced": random.choice([True, False]),
                "extra_fields": {},
            }
        )
        data_for_test = MockDTO(test_detections, pd.DataFrame())
            
        can_predict, message = self.model.can_predict(data_for_test)

        self.assertFalse(can_predict)

        # No features With Atlas alerts
        test_detections = pd.DataFrame(
            {
                "candid": str(random.randint(1000000, 9000000)),
                "oid": "ZTFoid",
                "aid": "aid",
                "tid": "atlas",
                "mjd": random.uniform(59000, 60000),
                "sid": "ZTF",
                "fid": "g",
                "pid": random.randint(1000000, 9000000),
                "ra": random.uniform(-90, 90),
                "e_ra": random.uniform(-90, 90),
                "dec": random.uniform(-90, 90),
                "e_dec": random.uniform(-90, 90),
                "mag": random.uniform(15, 20),
                "e_mag": random.uniform(0, 1),
                "mag_corr": random.uniform(15, 20),
                "e_mag_corr": random.uniform(0, 1),
                "e_mag_corr_ext": random.uniform(0, 1),
                "isdiffpos": random.choice([-1, 1]),
                "corrected": random.choice([True, False]),
                "dubious": random.choice([True, False]),
                "has_stamp": random.choice([True, False]),
                "stellar": random.choice([True, False]),
                "new": random.choice([True, False]),
                "forced": random.choice([True, False]),
                "extra_fields": {},
            },{
                "candid": str(random.randint(1000000, 9000000)),
                "oid": "ZTFoid2",
                "aid": "aid2",
                "tid": "atlas",
                "mjd": random.uniform(59000, 60000),
                "sid": "ZTF",
                "fid": "g",
                "pid": random.randint(1000000, 9000000),
                "ra": random.uniform(-90, 90),
                "e_ra": random.uniform(-90, 90),
                "dec": random.uniform(-90, 90),
                "e_dec": random.uniform(-90, 90),
                "mag": random.uniform(15, 20),
                "e_mag": random.uniform(0, 1),
                "mag_corr": random.uniform(15, 20),
                "e_mag_corr": random.uniform(0, 1),
                "e_mag_corr_ext": random.uniform(0, 1),
                "isdiffpos": random.choice([-1, 1]),
                "corrected": random.choice([True, False]),
                "dubious": random.choice([True, False]),
                "has_stamp": random.choice([True, False]),
                "stellar": random.choice([True, False]),
                "new": random.choice([True, False]),
                "forced": random.choice([True, False]),
                "extra_fields": {},
            }
        )
        data_for_test = MockDTO(test_detections, pd.DataFrame())
            
        can_predict, message = self.model.can_predict(data_for_test)

        self.assertFalse(can_predict)
        
        # With features and Atlas detections
        data_for_test = MockDTO(test_detections, self.test_features)
            
        can_predict, message = self.model.can_predict(data_for_test)

        self.assertFalse(can_predict)

        # With features No Alerts
        data_for_test = MockDTO(pd.DataFrame(), self.test_features)
            
        can_predict, message = self.model.can_predict(data_for_test)

        self.assertFalse(can_predict)
