#!/usr/bin/env python

from lc_classifier.features.turbofats.FeatureSpace import FeatureSpace
import numpy as np
import pandas as pd
import pytest

np.random.seed(1)

@pytest.fixture
def white_noise():
    time = np.random.choice(15_000, 10_000, replace=False)
    time.sort()
    mag = np.random.normal(loc=0.0, scale=1, size=len(time))
    error = np.random.normal(loc=0.01, scale=0.8, size=len(time))

    lc = np.stack([mag, time, error], axis=1)
    lightcurve_df = pd.DataFrame(
        data=lc,
        columns=[
            'magnitude',
            'time',
            'error'
        ],
        index=['dummy_id']*len(lc)
    )
    return lightcurve_df


@pytest.fixture
def uniform_lc():
    mjd_uniform = np.arange(1_000_000)
    data_uniform = np.random.uniform(size=1_000_000)
    lc = np.stack([data_uniform, mjd_uniform], axis=1)
    lightcurve_df = pd.DataFrame(
        data=lc,
        columns=[
            'magnitude',
            'time'
        ],
        index=['dummy_id']*len(lc)
    )

    return lightcurve_df


@pytest.fixture
def random_walk():
    N = 10_000
    alpha = 1.
    sigma = 0.5
    data_rw = np.zeros([N, 1])
    data_rw[0] = 1
    time_rw = range(1, N)
    for t in time_rw:
        data_rw[t] = alpha * data_rw[t - 1] + np.random.normal(loc=0.0, scale=sigma)
    time_rw = np.array(range(0, N)) + 1 * np.random.uniform(size=N)
    data_rw = data_rw.squeeze()
    lc = np.stack([data_rw, time_rw], axis=1)
    lightcurve_df = pd.DataFrame(
        data=lc,
        columns=[
            'magnitude',
            'time'
        ],
        index=['dummy_id']*len(lc)
    )
    return lightcurve_df


def test_Beyond1Std(white_noise):
    feature_name = 'Beyond1Std'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= 0.30 and value <= 0.40)


def test_Mean(white_noise):
    feature_name = 'Mean'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= -0.1 and value <= 0.1)


def test_Con(white_noise):
    feature_name = 'Con'
    feature_space = FeatureSpace(
        feature_list=[feature_name],
        extra_arguments={
            'Con': {
                'consecutiveStar': 1
            }
        }
    )
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= 0.04 and value <= 0.05)


def test_Eta_e(white_noise):
    feature_name = 'Eta_e'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]
    
    assert (value >= 1.9 and value <= 2.1)


def test_FluxPercentile(white_noise):
    # data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

    feature_space = FeatureSpace(
        feature_list=[
            'FluxPercentileRatioMid20', 
            'FluxPercentileRatioMid35', 
            'FluxPercentileRatioMid50',
            'FluxPercentileRatioMid65', 
            'FluxPercentileRatioMid80'
        ]
    )
    features = feature_space.calculate_features(white_noise)
    p20 = features['FluxPercentileRatioMid20'][0]
    p35 = features['FluxPercentileRatioMid35'][0]
    p50 = features['FluxPercentileRatioMid50'][0]
    p65 = features['FluxPercentileRatioMid65'][0]
    p80 = features['FluxPercentileRatioMid80'][0]
    
    assert (p20 >= 0.145 and p20 <= 0.160)
    assert (p35 >= 0.260 and p35 <= 0.290)
    assert (p50 >= 0.350 and p50 <= 0.450)
    assert (p65 >= 0.540 and p65 <= 0.580)
    assert (p80 >= 0.760 and p80 <= 0.800)


def test_LinearTrend(white_noise):
    feature_name = 'LinearTrend'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= -0.1 and value <= 0.1)


def test_Meanvariance(uniform_lc):
    feature_name = 'Meanvariance'
    feature_space = FeatureSpace(
        feature_list=[feature_name],
        data_column_names=uniform_lc.columns
    )
    features = feature_space.calculate_features(uniform_lc)
    value = features[feature_name][0]

    assert (value >= 0.575 and value <= 0.580)


def test_MedianAbsDev(white_noise):
    feature_name = 'MedianAbsDev'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= 0.630 and value <= 0.700)


def test_PairSlopeTrend(white_noise):
    feature_name = 'PairSlopeTrend'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= -0.25 and value <= 0.25)


def test_Q31(white_noise):
    feature_name = 'Q31'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= 1.30 and value <= 1.38)


def test_Rcs(white_noise):
    feature_name = 'Rcs'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= 0 and value <= 0.1)


def test_Skew(white_noise):
    feature_name = 'Skew'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= -0.1 and value <= 0.1)


def test_SmallKurtosis(white_noise):
    feature_name = 'SmallKurtosis'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= -0.2 and value <= 0.2)


def test_Std(white_noise):
    feature_name = 'Std'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= 0.9 and value <= 1.1)


def test_Stetson(white_noise):
    feature_space = FeatureSpace(feature_list=['SlottedA_length', 'StetsonK', 'StetsonK_AC'])
    features = feature_space.calculate_features(white_noise)

    stetson_k = features['StetsonK'][0]
    stetson_k_ac = features['StetsonK_AC'][0]

    print(features)
    assert (stetson_k_ac >= 0.20 and stetson_k_ac <= 0.45)
    assert (stetson_k <= 0.1)


def test_Gskew(white_noise):
    feature_name = 'Gskew'
    feature_space = FeatureSpace(feature_list=[feature_name])
    features = feature_space.calculate_features(white_noise)
    value = features[feature_name][0]

    assert (value >= -0.2 and value <= 0.2)


def test_StructureFunction(random_walk):
    feature_space = FeatureSpace(
        feature_list=[
            'StructureFunction_index_21', 
            'StructureFunction_index_31',
            'StructureFunction_index_32'
        ],
        data_column_names=random_walk.columns
    )
    
    features = feature_space.calculate_features(random_walk)
    sf21 = features['StructureFunction_index_21'][0]
    sf31 = features['StructureFunction_index_31'][0]
    sf32 = features['StructureFunction_index_32'][0]
    
    assert (sf21 >= 1.520 and sf21 <= 2.067)
    assert (sf31 >= 1.821 and sf31 <= 3.162)
    assert (sf32 >= 1.243 and sf32 <= 1.562)
