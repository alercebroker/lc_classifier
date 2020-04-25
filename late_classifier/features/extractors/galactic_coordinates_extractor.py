from typing import List

from late_classifier.features.core.base import FeatureExtractor
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
import logging


class GalacticCoordinatesExtractor(FeatureExtractor):
    def get_features_keys(self) -> List[str]:
        return ['gal_b', 'gal_l']

    def get_required_keys(self) -> List[str]:
        return ['ra', 'dec']

    def _compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with detections of an object.
        kwargs Not required.

        Returns :class:pandas.`DataFrame`
        -------

        """
        index = detections.index[0]

        mean_ra = detections.ra.mean()
        mean_dec = detections.dec.mean()
        coordinates = SkyCoord(ra=mean_ra, dec=mean_dec, frame='icrs', unit='deg')
        galactic = coordinates.galactic
        np_galactic = np.array([[galactic.b.degree, galactic.l.degree]])
        df = pd.DataFrame(np_galactic, columns=self.get_features_keys(), index=[index])
        return df
