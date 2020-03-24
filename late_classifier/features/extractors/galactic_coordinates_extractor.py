from late_classifier.features.core.base import FeatureExtractor
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np


class GalacticCoordinatesExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.features_keys = ['gal_b', 'gal_l']

    def compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with detections of an object.
        kwargs Not required.

        Returns :class:pandas.`DataFrame`
        -------

        """
        radec_df = detections[['ra', 'dec']].groupby(level=0).mean()
        coordinates = SkyCoord(ra=radec_df['ra'], dec=radec_df['dec'], frame='icrs', unit='deg')
        galactic = coordinates.galactic
        np_galactic = np.stack((galactic.b.degree, galactic.l.degree), axis=-1)
        df = pd.DataFrame(np_galactic, index=radec_df.index, columns=['gal_b', 'gal_l'])
        df.index.name = 'oid'
        return df
