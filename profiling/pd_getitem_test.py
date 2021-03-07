import os
import pandas as pd

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(
    os.path.join(
        FILE_PATH,
        "../data_examples/"))

detections = pd.read_csv(
    os.path.join(EXAMPLES_PATH, '100_objects_detections_corr.csv'),
    index_col='objectId'
)


def a(d):
    mjd = detections.mjd.values
    mag = detections.magpsf.values
    err = detections.sigmapsf.values
    return


def b(d):
    data = detections[['mjd', 'magpsf', 'sigmapsf']].values
    mjd = data[:, 0]
    mag = data[:, 1]
    err = data[:, 2]
    return


def c(d):
    mjd = detections['mjd'].values
    mag = detections['magpsf'].values
    err = detections['sigmapsf'].values
    return


def fff():
    for i in range(100):
        a(detections)
        b(detections)
        c(detections)


# fff()


def df(d):
    mag = detections[['magpsf']].values
    return


def series(d):
    mag = detections['magpsf'].values
    return


def gg():
    for i in range(100):
        df(detections)  # 0.107 [s]
        series(detections)  # 0.0018 [s]


gg()
