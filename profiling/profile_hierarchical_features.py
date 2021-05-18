from lc_classifier.features import ZTFFeatureExtractor
import os
import logging
import pandas as pd
import time
import sys


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


class ProfiledHierarchicalFeatureExtractor(ZTFFeatureExtractor):
    def _compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        kwargs Possible non_detections :class:pandas.`DataFrame`
                        objects :class:pandas.`DataFrame`

        Returns DataFrame with all features
        -------

        """

        ti = time.time()
        required = ["non_detections", "objects"]
        for key in required:
            if key not in kwargs:
                raise Exception(f"HierarchicalFeaturesComputer requires {key} argument")
        objects = kwargs["objects"]
        detections = self.preprocessor.preprocess(detections, objects=objects)
        has_enough_alerts = self.get_enough_alerts_mask(detections)
        too_short_oids = has_enough_alerts[~has_enough_alerts]
        too_short_features = pd.DataFrame(index=too_short_oids.index)
        detections = detections.loc[has_enough_alerts]
        detections = detections.sort_values("mjd")
        non_detections = kwargs["non_detections"]

        if len(non_detections) == 0:
            non_detections = pd.DataFrame(columns=["mjd", "fid", "diffmaglim"])

        features = []
        shared_data = dict()
        grouped_detections = detections.groupby(level=0)
        t_elapsed = time.time() - ti
        logging.info(f"profiling:Data preprocessing:{t_elapsed:5f}")

        for ex in self.extractors:
            ti = time.time()
            df = ex.compute_features(
                grouped_detections,
                non_detections=non_detections,
                shared_data=shared_data)
            t_elapsed = time.time() - ti
            logging.info(f"profiling:{ex.__class__}:{t_elapsed:5f}")
            features.append(df)
        df = pd.concat(features, axis=1, join="inner")
        df = pd.concat([df, too_short_features], axis=0, join="outer", sort=True)
        return df


def get_trimmed_curves(lc_len):
    t_det = []
    t_non_det = []
    t_obj = []
    for oid in detections.index.unique():
        oid_det = detections.loc[oid]
        if len(oid_det) < lc_len:
            continue
        oid_det = oid_det.sort_values('mjd')
        date = oid_det['mjd'].iloc[lc_len - 1]

        t_det.append(oid_det.iloc[:lc_len])
        if oid in non_detections.index.unique():
            oid_non_det = non_detections.loc[[oid]]
            oid_non_det = oid_non_det[oid_non_det['mjd'] < date]
            t_non_det.append(oid_non_det)

        t_obj.append(objects.loc[[oid]])

    t_det = pd.concat(t_det)
    t_non_det = pd.concat(t_non_det)
    t_obj = pd.concat(t_obj)

    return t_det, t_non_det, t_obj


def run_experiment(lc_len):
    custom_hierarchical_extractor = ProfiledHierarchicalFeatureExtractor(bands=[1, 2])

    t_det, t_non_det, t_obj = get_trimmed_curves(lc_len)
    logging.info(f"profiling:Experiment with lc_len {lc_len}, n_curves {len(t_det.index.unique())}")
    features_df = custom_hierarchical_extractor.compute_features(
        detections=t_det,
        non_detections=t_non_det,
        objects=t_obj
    )


mode = sys.argv[1]
if mode == 'many-lengths':
    logging.basicConfig(filename=sys.argv[2],  level=logging.INFO)

    for l in [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
        run_experiment(l)

elif mode == 'loop':
    while True:
        run_experiment(100)

else:
    print("Please enter mode: many-lengths, loop")
