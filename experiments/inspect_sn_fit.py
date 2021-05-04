import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lc_classifier.features.extractors.sn_parametric_model_computer import SNModelScipy
from lc_classifier.utils import mag_to_flux

detections = pd.read_pickle(sys.argv[1])
labels = pd.read_pickle(sys.argv[2])
fitted_params = pd.read_pickle(sys.argv[3])

available_sne = labels[labels.classALeRCE.isin(['SNIa', 'SNIbc', 'SNII', 'SLSN'])]
available_sne_index = available_sne.index.intersection(fitted_params.index)
available_sne = available_sne.loc[available_sne_index]

print(len(available_sne_index), available_sne_index)

sn_model = SNModelScipy()

oid = available_sne_index.values[0]


def plot_oid(oid):
    for fid in [1, 2]:
        plt.subplot(1, 2, fid)
        source = detections.loc[oid]
        source = source.sort_values('mjd')
        source = source[source.fid == fid]
        times = source.mjd
        times = times - np.min(times)
        mag = source.magpsf_corr
        flux = mag_to_flux(mag)

        parameters = fitted_params.loc[oid]
        if fid == 1:
            parameters = parameters.values.flatten()[:6]
        elif fid == 2:
            parameters = parameters.values.flatten()[7:13]
        model_times = np.linspace(0, np.max(times), 100)
        model_flux = sn_model.model(
            model_times,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
            parameters[5])

        plt.plot(times, flux, 'b+', label='real')
        plt.plot(model_times, model_flux, 'r-', label='model')
        plt.title(str(fid))
    plt.show()

plot_oid(oid)
