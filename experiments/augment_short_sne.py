import sys
import pandas as pd
from lc_classifier.augmentation.simple_augmentation import ShortTransientAugmenter


detections = pd.read_pickle(sys.argv[1])
non_detections = pd.read_pickle(sys.argv[2])
labels = pd.read_pickle(sys.argv[3])

detections = detections.loc[labels.index.unique()]
aux_index = labels.index.unique().intersection(
    non_detections.index.unique())
non_detections = non_detections.loc[aux_index]

augmenter = ShortTransientAugmenter()
new_detections, new_non_detections, new_labels = augmenter.augment(
    detections,
    non_detections,
    labels,
    multiplier=5
)

new_detections.to_pickle('short_sne_detections.pkl')
new_non_detections.to_pickle('short_sne_non_detections.pkl')
new_labels.to_pickle('short_sne_labels.pkl')
