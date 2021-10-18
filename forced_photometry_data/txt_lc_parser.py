import numpy as np
import pandas as pd
import os


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


def build_forced_detections_df():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    example_txt_files = [
        'forcedphotometry_req00010458_lc.txt',
        'forcedphotometry_req00010459_lc.txt',
        'forcedphotometry_req00010550_lc.txt',
        'forcedphotometry_req00012210_lc_ZTF20acfkyll.txt',
        'forcedphotometry_req00012234_lc_ZTF21aaqwjlz.txt'
    ]

    dataframes = []
    for i, filename in enumerate(example_txt_files):
        absolute_filepath = os.path.join(this_dir, filename)
        df = parse_lightcurve_txt(absolute_filepath)
        df['oid'] = 'ZTF_FAKE_OID' + str(i).zfill(2)
        df.set_index('oid', inplace=True)
        dataframes.append(df)
    df = pd.concat(dataframes, axis=0)
    df['fid'] = df['filter'].apply(lambda name: filter_name_to_int[name])
    df['mjd'] = df['jd'] - 2400000.5
    df['difference_flux'] = df['forcediffimflux']
    df['difference_flux_error'] = df['forcediffimfluxunc']

    nearestrefflux = 10 ** (0.4 * (df['zpdiff'] - df['nearestrefmag']))
    nearestreffluxunc = df['nearestrefmagunc'] * nearestrefflux / 1.0857
    flux = df['forcediffimflux'] + nearestrefflux
    with np.errstate(invalid='ignore'):
        mag = df['zpdiff'] - 2.5 * np.log10(flux)
    fluxunc = np.sqrt(
        df['forcediffimfluxunc'] ** 2 + nearestreffluxunc ** 2
    )
    snr_tot = flux / fluxunc
    err = 1.0857 / snr_tot

    df['magpsf_ml'] = mag
    df['sigmapsf_ml'] = err
    return df


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
