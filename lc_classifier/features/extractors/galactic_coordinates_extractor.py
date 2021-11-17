from typing import Tuple
from functools import lru_cache

from ..core.base import FeatureExtractor
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np


class GalacticCoordinatesExtractor(FeatureExtractor):
    def __init__(self, from_metadata=False):
        super(GalacticCoordinatesExtractor, self).__init__()
        self.from_metadata = from_metadata

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'gal_b', 'gal_l'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        if self.from_metadata:
            return ()
        else:
            return 'ra', 'dec'

    def compute_from_metadata(self, detections, metadata):
        coordinates = SkyCoord(
            ra=metadata['ra'],
            dec=metadata['dec'],
            frame='icrs',
            unit='deg')
        galactic = coordinates.galactic
        np_galactic = np.stack((galactic.b.degree, galactic.l.degree), axis=-1)
        galactic_coordinates_df = pd.DataFrame(
            np_galactic,
            index=metadata.index,
            columns=['gal_b', 'gal_l'])
        galactic_coordinates_df.index.name = 'oid'
        det_oids = detections.index.unique()
        galactic_coordinates_df = galactic_coordinates_df.loc[det_oids]
        return galactic_coordinates_df

    def compute_from_detections(self, detections):
        radec_df = detections[['ra', 'dec']].groupby(level=0).mean()
        coordinates = SkyCoord(
            ra=radec_df.values[:, 0],
            dec=radec_df.values[:, 1],
            frame='icrs',
            unit='deg')
        galactic = coordinates.galactic
        np_galactic = np.stack((galactic.b.degree, galactic.l.degree), axis=-1)
        galactic_coordinates_df = pd.DataFrame(
            np_galactic,
            index=radec_df.index,
            columns=['gal_b', 'gal_l'])
        galactic_coordinates_df.index.name = 'oid'
        return galactic_coordinates_df

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
        if self.from_metadata:
            return self.compute_from_metadata(detections, kwargs['metadata'])
        else:
            return self.compute_from_detections(detections)
