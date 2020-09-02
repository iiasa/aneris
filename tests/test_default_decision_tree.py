import pandas as pd
import pytest

from aneris import harmonize

import pandas.util.testing as pdt


def make_index(length, gas='CH4', sector='Energy'):
    return pd.MultiIndex.from_product(
        [["region_{i}" for i in range(length)], [gas], [sector]],
        names=["region", "gas", "sector"]
    )


@pytest.fixture
def index1():
    return make_index(1)


def test_hist_zero(index1):
    hist = pd.DataFrame({'2015': [0]}, index1)
    df = pd.DataFrame({'2015': [1.]}, index1)

    obs, diags = harmonize.default_methods(hist, df, '2015')

    exp = pd.Series(['hist_zero'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_model_zero(index1):
    hist = pd.DataFrame({'2015': [1.]}, index1)
    df = pd.DataFrame({'2015': [0.]}, index1)

    obs, diags = harmonize.default_methods(hist, df, '2015')

    exp = pd.Series(['model_zero'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch1(index1):
    hist = pd.DataFrame({'2015': [1.]}, index1)
    df = pd.DataFrame(
        {'2015': [0.], '2020': [-1.]},
        index1
    )

    obs, diags = harmonize.default_methods(hist, df, '2015')
    exp = pd.Series(['reduce_offset_2080'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)

    obs, diags = harmonize.default_methods(hist, df, '2015',
                                           offset_method='reduce_offset_2050')
    exp = pd.Series(['reduce_offset_2050'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch2(index1):
    hist = pd.DataFrame({'2015': [1.]}, index1)
    df = pd.DataFrame(
        {'2015': [0.], '2020': [1.]},
        index1
    )

    obs, diags = harmonize.default_methods(hist, df, '2015')
    exp = pd.Series(['constant_offset'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch3(index1):
    hist = pd.DataFrame(
        {'2015': [1.]},
        index1
    )
    df = pd.DataFrame(
        {'2015': [1.001], '2020': [-1.001]},
        index1
    )

    obs, diags = harmonize.default_methods(hist, df, '2015')
    exp = pd.Series(['reduce_ratio_2080'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)

    obs, diags = harmonize.default_methods(hist, df, '2015',
                                           ratio_method='reduce_ratio_2050')
    exp = pd.Series(['reduce_ratio_2050'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch4(index1):
    hist = pd.DataFrame({'2015': [1.]}, index1)
    df = pd.DataFrame(
        {'2015': [5.001], '2020': [-1.]},
        index1
    )

    obs, diags = harmonize.default_methods(hist, df, '2015')

    exp = pd.Series(['reduce_ratio_2100'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch5(index1):
    hist = pd.DataFrame({'2015': [1.]}, index1)
    df = pd.DataFrame(
        {'2015': [5.001], '2020': [1.]},
        index1
    )

    obs, diags = harmonize.default_methods(hist, df, '2015')

    exp = pd.Series(['constant_ratio'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch6(index1):
    hist = pd.DataFrame(
        {
            '2000': [1.],
            '2005': [1000.],
            '2010': [1.],
            '2015': [100.],
        },
        index1
    )
    df = pd.DataFrame(
        {
            '2015': [5.001],
            '2020': [1.],
        },
        index1
    )

    obs, diags = harmonize.default_methods(hist, df, '2015')
    print(diags)

    exp = pd.Series(['reduce_offset_2150_cov'], index1, name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)
