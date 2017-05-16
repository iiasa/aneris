import pandas as pd

from aneris import harmonize

import pandas.util.testing as pdt


def test_hist_zero():
    hist = pd.DataFrame({'2015': [0]})
    df = pd.DataFrame({'2015': [1.]})

    obs, diags = harmonize.default_methods(hist, df, '2015')

    exp = pd.Series(['hist_zero'], name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_model_zero():
    hist = pd.DataFrame({'2015': [1.]})
    df = pd.DataFrame({'2015': [0.]})

    obs, diags = harmonize.default_methods(hist, df, '2015')

    exp = pd.Series(['model_zero'], name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch1():
    hist = pd.DataFrame({'2015': [1.]})
    df = pd.DataFrame({
        '2015': [0.],
        '2020': [-1.],
    })

    obs, diags = harmonize.default_methods(hist, df, '2015')
    exp = pd.Series(['reduce_offset_2080'], name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch2():
    hist = pd.DataFrame({'2015': [1.]})
    df = pd.DataFrame({
        '2015': [0.],
        '2020': [1.],
    })

    obs, diags = harmonize.default_methods(hist, df, '2015')
    exp = pd.Series(['constant_offset'], name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch3():
    hist = pd.DataFrame({
        '2015': [1.],
    })
    df = pd.DataFrame({
        '2015': [1.001],
        '2020': [-1.001],
    })

    obs, diags = harmonize.default_methods(hist, df, '2015')
    exp = pd.Series(['reduce_ratio_2080'], name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch4():
    hist = pd.DataFrame({'2015': [1.]})
    df = pd.DataFrame({
        '2015': [5.001],
        '2020': [-1.],
    })

    obs, diags = harmonize.default_methods(hist, df, '2015')

    exp = pd.Series(['reduce_ratio_2100'], name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch5():
    hist = pd.DataFrame({'2015': [1.]})
    df = pd.DataFrame({
        '2015': [5.001],
        '2020': [1.],
    })

    obs, diags = harmonize.default_methods(hist, df, '2015')

    exp = pd.Series(['constant_ratio'], name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch6():
    hist = pd.DataFrame({
        '2000': [1.],
        '2005': [1000.],
        '2010': [1.],
        '2015': [100.],
    })
    df = pd.DataFrame({
        '2015': [5.001],
        '2020': [1.],
    })

    obs, diags = harmonize.default_methods(hist, df, '2015')
    print(diags)

    exp = pd.Series(['reduce_offset_2150_cov'], name='methods')
    pdt.assert_series_equal(exp, obs, check_names=False)
