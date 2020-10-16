from lc_classifier.features import CustomHierarchicalExtractor
import os
import pandas as pd

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(
    os.path.join(
        FILE_PATH,
        "../data_examples/"))

detections = pd.read_csv(
    os.path.join(EXAMPLES_PATH, '100_objects_detections_corr.csv'),
    index_col='objectId'
)

non_detections = pd.read_csv(
    os.path.join(EXAMPLES_PATH, '100_objects_non_detections.csv'),
    index_col='objectId'
)
non_detections["mjd"] = non_detections.jd - 2400000.5

objects = pd.read_csv(
    os.path.join(EXAMPLES_PATH, '100_objects.csv'),
    index_col='objectId'
)

detections.index.name = "oid"
non_detections.index.name = "oid"
objects.index.name = "oid"

custom_hierarchical_extractor = CustomHierarchicalExtractor(bands=[1, 2])

features_df = custom_hierarchical_extractor.compute_features(
    detections=detections,
    non_detections=non_detections,
    objects=objects
)
