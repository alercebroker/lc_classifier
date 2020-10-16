import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sn_params = pd.read_pickle('sn_param_features.pkl')
labels = pd.read_pickle('training_labels.pkl')

fit_errors = sn_params[['sn_model_fit_error_1', 'sn_model_fit_error_2']].dropna()
fit_labels_index = labels.index.intersection(fit_errors.index)
fit_errors = fit_errors.loc[fit_labels_index]
fit_labels = labels.loc[fit_labels_index]

is_sn = fit_labels.classALeRCE.isin(['SNIa', 'SNII', 'SNIbc', 'SLSN'])

sne_sample = fit_errors[is_sn].copy()#.sample(n=400)
non_sne_sample = fit_errors[~is_sn].sample(n=1000)
sne_sample['color'] = 1.0
non_sne_sample['color'] = 0.0

sample = pd.concat([sne_sample, non_sne_sample], axis=0)

plt.scatter(
    sample['sn_model_fit_error_1'],
    sample['sn_model_fit_error_2'],
    c=sample['color'],
    alpha=0.7,
    edgecolors='k'
)
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.gca().set_ylim([np.min(sample['sn_model_fit_error_2']), np.max(sample['sn_model_fit_error_2'])])
plt.gca().set_xlim([np.min(sample['sn_model_fit_error_1']), np.max(sample['sn_model_fit_error_1'])])
plt.show()
