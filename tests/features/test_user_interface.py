import os
import unittest
from lc_classifier.features import MHPSExtractor, PeriodExtractor, GPDRWExtractor
from lc_classifier.features import FoldedKimExtractor, GalacticCoordinatesExtractor
from lc_classifier.features import HarmonicsExtractor, IQRExtractor
from lc_classifier.features import PowerRateExtractor, SNFeaturesPhaseIIExtractor
from lc_classifier.features import SPMExtractorPhaseII, TurboFatsFeatureExtractor

from lc_classifier.utils import LightcurveBuilder, mag_to_flux
from lc_classifier.features import FeatureExtractorComposer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_sn2012aw():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        this_dir,
        "../../data_examples/SN2012aw"
    )
    oid = 'SN2012aw'
    dfs = []
    for filename in os.listdir(data_dir):
        if 'info' in filename:
            continue
        band = '.'.join(filename.split('.')[1:])[4:]
        loaded_csv = pd.read_csv(
            os.path.join(data_dir, filename),
            sep="\s{2,}",
            header=3
        )
        loaded_csv['band'] = band
        dfs.append(loaded_csv)
    lightcurve = pd.concat(dfs, axis=0)
    lightcurve['oid'] = oid
    lightcurve.set_index('oid', inplace=True)
    return lightcurve


def plot_sn2012aw():
    lightcurve = load_sn2012aw()
    lightcurve.rename(
        columns={
            'MJD': 'time',
            'Mag': 'magnitude',
            'DMag': 'error'
        },
        inplace=True
    )
    print(lightcurve)
    for band in lightcurve['band'].unique():
        band_lc = lightcurve[lightcurve['band'] == band]
        plt.scatter(band_lc.time, band_lc.magnitude)
    plt.gca().invert_yaxis()
    plt.show()


class TestUserInterface(unittest.TestCase):
    def test_lightcurve_builder(self):
        oid = 'dummy_oid'
        lc_builder = LightcurveBuilder(oid)
        for band in ['g', 'r', 'i']:
            n_obs_band = np.random.randint(10, 30)
            time = np.random.rand(n_obs_band) * 500
            magnitude = np.random.randn(n_obs_band) + 17.0
            error = np.random.randn(n_obs_band) ** 2 * 2.0

            lc_builder.add_band(
                band,
                time,
                magnitude,
                error)
        lc_dataframe = lc_builder.build_dataframe()

    def test_compose_extractor(self):
        lightcurve = load_sn2012aw()
        lightcurve.rename(
            columns={
                'MJD': 'time',
                'Mag': 'magnitude',
                'DMag': 'error'
            },
            inplace=True
        )

        lightcurve['ra'] = np.random.randn(len(lightcurve)) + 30.0
        lightcurve['dec'] = np.random.randn(len(lightcurve)) + 30.0

        lightcurve['difference_flux'] = mag_to_flux(lightcurve['magnitude'].values)
        lightcurve['difference_flux_error'] = (
            mag_to_flux(lightcurve['magnitude'].values - lightcurve['error'].values)
            - lightcurve['difference_flux'].values
        )

        bands = lightcurve.band.unique()
        feature_extractor = FeatureExtractorComposer(
            [
                MHPSExtractor(bands),
                PeriodExtractor(bands),
                GPDRWExtractor(bands),
                FoldedKimExtractor(bands),
                GalacticCoordinatesExtractor(),
                HarmonicsExtractor(bands),
                IQRExtractor(bands),
                PowerRateExtractor(bands),
                SNFeaturesPhaseIIExtractor(bands),
                SPMExtractorPhaseII(bands),
                TurboFatsFeatureExtractor(bands)
            ]
        )

        features_df = feature_extractor.compute_features(lightcurve)


if __name__ == '__main__':
    unittest.main()
