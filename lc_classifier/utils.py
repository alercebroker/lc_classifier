import numba
import numpy as np
import pandas as pd


class LightcurveBuilder:
    def __init__(self, oid: str):
        self.oid = oid
        self.band_dataframes = []

    def add_band(
            self,
            band: str,
            time: np.array,
            magnitude: np.array,
            error: np.array):
        band_dictionary = {
            'time': time,
            'magnitude': magnitude,
            'error': error
        }
        band_dataframe = pd.DataFrame.from_dict(
            band_dictionary
        )
        band_dataframe['band'] = band
        band_dataframe['oid'] = self.oid
        self.band_dataframes.append(band_dataframe)

    def build_dataframe(self) -> pd.DataFrame:
        df = pd.concat(self.band_dataframes, axis=0)
        df.set_index('oid', inplace=True)
        return df


@numba.jit(nopython=True)
def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True


@numba.jit(nopython=True)
def first_occurrence(a: np.ndarray, value):
    for i, x in enumerate(a):
        if x == value:
            return i
    return -1


def mag_to_flux(mag):
    """Converts a list of magnitudes into flux."""
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)
