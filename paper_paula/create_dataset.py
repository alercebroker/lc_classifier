import numpy as np
import pandas as pd


detections = pd.read_csv(
    'detections.csv',
    index_col='oid',
    na_values=['Infinity'])

non_detections = pd.read_csv(
    'non_detections.csv',
    index_col='oid',
    na_values=['Infinity'])

labels = pd.read_csv(
    'dfcrossmatches_prioritized_v5.1.csv',
    index_col='oid',
    na_values=['Infinity']
)

labeled_detections = detections.loc[labels.index]
labeled_non_detections = non_detections.loc[labels.index]
