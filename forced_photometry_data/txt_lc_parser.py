import numpy as np
import pandas as pd


def parse_lightcurve_txt(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=' ',
        comment='#',
        skipinitialspace=True,
    )
    columns = [name.rstrip(',') for name in df.columns.values]
    df.columns = columns
    df.set_index('index', inplace=True)
    return df


filter_name_to_int = {
    'ZTF_g': 1,
    'ZTF_r': 2,
    'ZTF_i': 3
}


if __name__ == '__main__':
    lightcurve_df = parse_lightcurve_txt('forcedphotometry_req00010458_lc.txt')
    print(lightcurve_df.head())

    import matplotlib.pyplot as plt
    time = lightcurve_df['jd']
    time -= time.min()

    # nearestrefflux = 10**(0.4*(lightcurve_df['zpdiff'] - lightcurve_df['nearestrefmag']))
    # flux = lightcurve_df['forcediffimflux'] + nearestrefflux
    # mag = lightcurve_df['zpdiff'] - 2.5*np.log10(flux)

    diff_flux = lightcurve_df['forcediffimflux']
    plt.scatter(
        time,
        diff_flux,
        c=[filter_name_to_int[name] for name in lightcurve_df['filter']]
    )
    # plt.gca().invert_yaxis()
    plt.xlabel('days')
    plt.ylabel('magnitude')
    plt.title('ZTF21aaomuka difference-image flux')
    plt.show()
