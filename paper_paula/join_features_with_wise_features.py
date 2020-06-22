import sys
import pandas as pd


features = pd.read_pickle(sys.argv[1])
wise_features = pd.read_pickle(sys.argv[2])

new_features = pd.merge(features, wise_features, how='left', left_index=True, right_index=True)

new_features.to_pickle(sys.argv[3])
