import sys
import pandas as pd


bands = pd.read_pickle(sys.argv[1])

bands_original_columms = bands.columns

bands['W1-W2'] = bands['W1'] - bands['W2']
bands['W2-W3'] = bands['W2'] - bands['W3']
bands['g-W2'] = bands['g'] - bands['W2']
bands['g-W3'] = bands['g'] - bands['W3']  # redundant?
bands['r-W2'] = bands['r'] - bands['W2']
bands['r-W3'] = bands['r'] - bands['W3']  # redundant?

new_columns = [col for col in bands.columns if col not in bands_original_columms]

bands[new_columns].to_pickle(sys.argv[2])
