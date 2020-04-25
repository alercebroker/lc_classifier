from late_classifier.features.custom import CustomHierarchicalExtractor
import os
import pandas as pd
import cProfile


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(
    os.path.join(
        FILE_PATH,
        "../late_classifier/features/tests/data"))

detections = pd.read_csv(
    os.path.join(EXAMPLES_PATH, 'ZTF18aazsabq_det.csv'),
    index_col='oid'
)

non_detections = pd.read_csv(
    os.path.join(EXAMPLES_PATH, 'ZTF18aazsabq_nondet.csv'),
    index_col='oid'
)

custom_hierarchical_extractor = CustomHierarchicalExtractor(bands=[1, 2])

# Run once to avoid measuring jit-compilation cost
_ = custom_hierarchical_extractor.compute_features(
    detections=detections,
    non_detections=non_detections
)

profiler = cProfile.Profile()
profiler.enable()
features_df = custom_hierarchical_extractor.compute_features(
    detections=detections,
    non_detections=non_detections
)
profiler.disable()
profiler.dump_stats(
    os.path.join(
        FILE_PATH,
        'hierarchical_features.profile'
    )
)
