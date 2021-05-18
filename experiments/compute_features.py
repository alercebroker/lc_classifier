import sys
import numpy as np
import pandas as pd
from lc_classifier.features import ZTFFeatureExtractor
from joblib import Parallel, delayed


def extract_features(process_id, detections_df, non_detections_df):
    hierarchical_extractor = ZTFFeatureExtractor([1, 2])
    features = hierarchical_extractor.compute_features(
        detections_df, non_detections=non_detections_df)
    return features


batch_size = 200

detections = pd.read_pickle(sys.argv[1])
non_detections = pd.read_pickle(sys.argv[2])
non_detections_oids = non_detections.index.unique()

lc_idxs = detections.index.unique()
n_batches = int(np.ceil(len(lc_idxs)/batch_size))
print("Mini batch size: %d, Number of mini batches: %d" % (batch_size, n_batches))

JUST_ONE_BATCH = True
if JUST_ONE_BATCH:
    k = 0
    batch_size = 10000
    det_batch_oids = lc_idxs[k * batch_size: (k+1) * batch_size]
    non_det_batch_oids = non_detections_oids.intersection(det_batch_oids)  # to avoid KeyError
    features = extract_features(
        k,
        detections.loc[det_batch_oids],
        non_detections.loc[non_det_batch_oids])
    print(features)
    features.to_pickle(sys.argv[3])
    exit()

tasks = []
for k in range(n_batches):
    det_batch_oids = lc_idxs[k * batch_size: (k+1) * batch_size]
    non_det_batch_oids = non_detections_oids.intersection(det_batch_oids)  # to avoid KeyError
    tasks.append(
        delayed(extract_features)(
            k,
            detections.loc[det_batch_oids],
            non_detections.loc[non_det_batch_oids]))

tasks_ans = Parallel(n_jobs=12, verbose=11, backend="loky")(tasks)
features = pd.concat(tasks_ans, axis=0)
features.to_pickle(sys.argv[3])
