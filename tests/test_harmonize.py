import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from aneris import harmonize, utils


nvals = 6


_df = (
    pd.DataFrame(
        {
            "gas": ["BC"] * nvals,
            "region": ["a"] * nvals,
            "unit": ["Mt"] * nvals,
            "sector": ["bar", "foo"] + [str(x) for x in range(nvals - 2)],
            "2010": [2, 1, 9000, 9000, 9000, 9000],
            "2015": [3, 2, 0.51, 9000, 9000, -90],
            "2040": [4.5, 1.5, 9000, 9000, 9000, 9000],
            "2060": [6, 1, 9000, 9000, 9000, 9000],
        }
    )
    .set_index(utils.df_idx)
    .sort_index()
)

_t_frac = lambda tf: (2040 - 2015) / float(tf - 2015)

_hist = (
    pd.DataFrame(
        {
            "gas": ["BC"] * nvals,
            "region": ["a"] * nvals,
            "unit": ["Mt"] * nvals,
            "sector": ["bar", "foo"] + [str(x) for x in range(nvals - 2)],
            "2010": [1.0, 0.34, 9000, 9000, 9000, 9000],
            "2015": [0.01, 1.0, 0.5, 2 * 8999.0 / 9, 3 * 8999.0, 8999.0],
        }
    )
    .set_index(utils.df_idx)
    .sort_index()
)

_methods = (
    pd.DataFrame(
        {
            "gas": _df.index.get_level_values("gas"),
            "sector": _df.index.get_level_values("sector"),
            "region": ["a"] * nvals,
            "method": ["constant_offset"] * nvals,
        }
    )
    .set_index(["region", "gas", "sector"])
    .sort_index()
)


def test_factors():
    df = _df.copy()
    hist = _hist.copy()
    obsoffset, obsratio = harmonize.harmonize_factors(
        df.copy(), hist.copy(), harmonize_year="2015"
    )
    # im lazy; test initially written when these were of length 2
    exp = np.array([0.01 - 3, -1.0])
    npt.assert_array_almost_equal(exp, obsoffset[-2:])
    exp = np.array([0.01 / 3, 0.5])
    npt.assert_array_almost_equal(exp, obsratio[-2:])


def test_harmonize_constant_offset():
    df = _df.copy()
    hist = _hist.copy()
    methods = _methods.copy()
    h = harmonize.Harmonizer(df, hist)
    res = h.harmonize(year="2015", overrides=methods["method"])

    # base year
    obs = res["2015"]
    exp = _hist["2015"]
    npt.assert_array_almost_equal(obs, exp)

    # future year
    obs = res["2060"]
    exp = _df["2060"] + (_hist["2015"] - _df["2015"])
    npt.assert_array_almost_equal(obs, exp)


def test_no_model():
    df = pd.DataFrame({"2015": [0]})
    hist = pd.DataFrame({"2015": [1.5]})
    obsoffset, obsratio = harmonize.harmonize_factors(
        df.copy(), hist.copy(), harmonize_year="2015"
    )
    exp = np.array([1.5])
    npt.assert_array_almost_equal(exp, obsoffset)
    exp = np.array([0])
    npt.assert_array_almost_equal(exp, obsratio)


def test_harmonize_constant_ratio():
    df = _df.copy()
    hist = _hist.copy()
    methods = _methods.copy()
    h = harmonize.Harmonizer(df, hist)
    methods["method"] = ["constant_ratio"] * nvals
    res = h.harmonize(year="2015", overrides=methods["method"])

    # base year
    obs = res["2015"]
    exp = _hist["2015"]
    npt.assert_array_almost_equal(obs, exp)

    # future year
    obs = res["2060"]
    exp = _df["2060"] * (_hist["2015"] / _df["2015"])
    npt.assert_array_almost_equal(obs, exp)


def test_harmonize_reduce_offset():
    df = _df.copy()
    hist = _hist.copy()
    methods = _methods.copy()
    h = harmonize.Harmonizer(df, hist)

    # this is bad, there should be a test for each case
    for tf in [2050, 2100, 2150]:
        print(tf)
        method = f"reduce_offset_{tf}"
        methods["method"] = [method] * nvals
        res = h.harmonize(year="2015", overrides=methods["method"])

        # base year
        obs = res["2015"]
        exp = _hist["2015"]
        npt.assert_array_almost_equal(obs, exp)

        # future year
        obs = res["2040"]
        exp = _df["2040"] + (1 - _t_frac(tf)) * (_hist["2015"] - _df["2015"])
        npt.assert_array_almost_equal(obs, exp)

        # future year
        if tf < 2060:
            obs = res["2060"]
            exp = _df["2060"]
            npt.assert_array_almost_equal(obs, exp)


def test_harmonize_reduce_ratio():
    df = _df.copy()
    hist = _hist.copy()
    methods = _methods.copy()
    h = harmonize.Harmonizer(df, hist)

    # this is bad, there should be a test for each case
    for tf in [2050, 2100, 2150]:
        print(tf)
        method = f"reduce_ratio_{tf}"
        methods["method"] = [method] * nvals
        res = h.harmonize(year="2015", overrides=methods["method"])

        # base year
        obs = res["2015"]
        exp = _hist["2015"]
        npt.assert_array_almost_equal(obs, exp)

        # future year
        obs = res["2040"]
        ratio = _hist["2015"] / _df["2015"]
        exp = _df["2040"] * (ratio + _t_frac(tf) * (1 - ratio))
        npt.assert_array_almost_equal(obs, exp)

        # future year
        if tf < 2060:
            obs = res["2060"]
            exp = _df["2060"]
            npt.assert_array_almost_equal(obs, exp)


@pytest.mark.xfail(reason="standard interfaces can't handle units")
def test_harmonize_reduce_ratio_different_units():
    df = _df.copy()
    hist = _hist.copy()
    hist /= 1000
    hist.index = hist.index.set_levels(["kt"], "units")

    methods = _methods.copy()
    h = harmonize.Harmonizer(df, hist)

    tf = 2050

    method = f"reduce_ratio_{tf}"
    methods["method"] = [method] * nvals
    res = h.harmonize(year="2015", overrides=methods["method"])

    # base year
    obs = res["2015"]
    exp = hist["2015"]
    # should come back with input units
    obs_units = obs.index.get_level_values("units")
    df_units = df.index.get_level_values("units")
    assert (obs_units == df_units).all()
    npt.assert_array_almost_equal(obs, exp)

    # future year
    obs = res["2040"]
    ratio = _hist["2015"] / _df["2015"]
    exp = _df["2040"] * (ratio + _t_frac(tf) * (1 - ratio))
    npt.assert_array_almost_equal(obs, exp)

    # future year
    if tf < 2060:
        obs = res["2060"]
        exp = _df["2060"]
        npt.assert_array_almost_equal(obs, exp)


def test_harmonize_mix():
    df = _df.copy()
    hist = _hist.copy()
    methods = _methods.copy()
    h = harmonize.Harmonizer(df, hist)
    methods["method"] = ["constant_offset"] * nvals
    res = h.harmonize(year="2015", overrides=methods["method"])

    # base year
    obs = res["2015"]
    exp = _hist["2015"]
    npt.assert_array_almost_equal(obs, exp)

    # future year
    obs = res["2060"][:2]
    exp = [
        _df["2060"][0] + (_hist["2015"][0] - _df["2015"][0]),
        _df["2060"][1] * (_hist["2015"][1] / _df["2015"][1]),
    ]
    npt.assert_array_almost_equal(obs, exp)


def test_harmonize_linear_interpolation():
    df = _df.copy()
    hist = _hist.copy()
    methods = _methods.copy()
    h = harmonize.Harmonizer(df, hist)
    methods["method"] = ["linear_interpolate_2060"] * nvals
    res = h.harmonize(year="2015", overrides=methods["method"])

    # base year
    obs = res["2015"]
    exp = _hist["2015"]
    npt.assert_array_almost_equal(obs, exp)

    # future year
    x1, x2, x = "2015", "2060", "2040"
    y1, y2 = _hist[x1], _df[x2]
    m = (y2 - y1) / (float(x2) - float(x1))
    b = y1 - m * float(x1)
    obs = res[x]
    exp = m * float(x) + b
    npt.assert_array_almost_equal(obs, exp)

    # year after interp
    obs = res["2060"]
    exp = _df["2060"]
    npt.assert_array_almost_equal(obs, exp)


@pytest.mark.ipopt
def test_harmonize_budget():
    df = _df.copy()
    hist = _hist.copy()
    methods = _methods.copy()

    h = harmonize.Harmonizer(df, hist)
    methods["method"] = "budget"
    res = h.harmonize(year="2015", overrides=methods["method"])

    # base year
    obs = res["2015"]
    exp = _hist["2015"]
    npt.assert_array_almost_equal(obs, exp)

    # carbon budget conserved
    def _carbon_budget(emissions):
        # trapezoid rule
        dyears = np.diff(emissions.columns.astype(int))
        emissions = emissions.to_numpy()
        demissions = np.diff(emissions, axis=1)

        budget = (dyears * (emissions[:, :-1] + demissions / 2)).sum(axis=1)
        return budget

    npt.assert_array_almost_equal(
        _carbon_budget(res),
        _carbon_budget(df) - _carbon_budget(hist.loc[:, "2010":"2015"]),
    )
