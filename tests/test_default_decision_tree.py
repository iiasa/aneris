import pandas as pd
import pandas.testing as pdt
import pytest

from aneris import harmonize


def make_index(length, gas="CH4", sector="Energy"):
    return pd.MultiIndex.from_product(
        [["region_{i}" for i in range(length)], [gas], [sector]],
        names=["region", "gas", "sector"],
    )


@pytest.fixture
def index1():
    return make_index(1)


@pytest.fixture
def index1_co2():
    return make_index(1, gas="CO2")


def test_hist_zero(index1):
    hist = pd.DataFrame({"2015": [0]}, index1)
    df = pd.DataFrame({"2015": [1.0]}, index1)

    obs, diags = harmonize.default_methods(hist, df, "2015")

    exp = pd.Series(["hist_zero"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_model_zero(index1):
    hist = pd.DataFrame({"2015": [1.0]}, index1)
    df = pd.DataFrame({"2015": [0.0]}, index1)

    obs, diags = harmonize.default_methods(hist, df, "2015")

    exp = pd.Series(["model_zero"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch1(index1):
    hist = pd.DataFrame({"2015": [1.0]}, index1)
    df = pd.DataFrame({"2015": [0.0], "2020": [-1.0]}, index1)

    obs, diags = harmonize.default_methods(hist, df, "2015")
    exp = pd.Series(["reduce_offset_2080"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)

    obs, diags = harmonize.default_methods(
        hist, df, "2015", offset_method="reduce_offset_2050"
    )
    exp = pd.Series(["reduce_offset_2050"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch2(index1):
    hist = pd.DataFrame({"2015": [1.0]}, index1)
    df = pd.DataFrame({"2015": [0.0], "2020": [1.0]}, index1)

    obs, diags = harmonize.default_methods(hist, df, "2015")
    exp = pd.Series(["constant_offset"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch3(index1):
    hist = pd.DataFrame({"2015": [1.0]}, index1)
    df = pd.DataFrame({"2015": [1.001], "2020": [-1.001]}, index1)

    obs, diags = harmonize.default_methods(hist, df, "2015")
    exp = pd.Series(["reduce_ratio_2080"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)

    obs, diags = harmonize.default_methods(
        hist, df, "2015", ratio_method="reduce_ratio_2050"
    )
    exp = pd.Series(["reduce_ratio_2050"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch4(index1):
    hist = pd.DataFrame({"2015": [1.0]}, index1)
    df = pd.DataFrame({"2015": [5.001], "2020": [-1.0]}, index1)

    obs, diags = harmonize.default_methods(hist, df, "2015")

    exp = pd.Series(["reduce_ratio_2100"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch5(index1):
    hist = pd.DataFrame({"2015": [1.0]}, index1)
    df = pd.DataFrame({"2015": [5.001], "2020": [1.0]}, index1)

    obs, diags = harmonize.default_methods(hist, df, "2015")

    exp = pd.Series(["constant_ratio"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_branch6(index1):
    hist = pd.DataFrame(
        {
            "2000": [1.0],
            "2005": [1000.0],
            "2010": [1.0],
            "2015": [100.0],
        },
        index1,
    )
    df = pd.DataFrame(
        {
            "2015": [5.001],
            "2020": [1.0],
        },
        index1,
    )

    obs, diags = harmonize.default_methods(hist, df, "2015")
    print(diags)

    exp = pd.Series(["reduce_offset_2150_cov"], index1, name="methods")
    pdt.assert_series_equal(exp, obs, check_names=False)


def test_custom_method_choice(index1, index1_co2):
    def method_choice(
        row,
        ratio_method="reduce_ratio_2080",
        offset_method=None,
        luc_method=None,
        luc_cov_threshold=None,
    ):
        return "budget" if row.gas == "CO2" else ratio_method

    # CH4
    hist_ch4 = pd.DataFrame({"2015": [1.0]}, index1)
    df_ch4 = pd.DataFrame({"2015": [1.0]}, index1)

    obs_ch4, _ = harmonize.default_methods(
        hist_ch4, df_ch4, "2015", method_choice=method_choice
    )

    exp_ch4 = pd.Series(["reduce_ratio_2080"], index1, name="methods")
    pdt.assert_series_equal(exp_ch4, obs_ch4, check_names=False)

    # CO2
    hist_co2 = pd.DataFrame({"2015": [1.0]}, index1_co2)
    df_co2 = pd.DataFrame({"2015": [1.0]}, index1_co2)

    obs_co2, _ = harmonize.default_methods(
        hist_co2, df_co2, "2015", method_choice=method_choice
    )

    exp_co2 = pd.Series(["budget"], index1_co2, name="methods")
    pdt.assert_series_equal(exp_co2, obs_co2, check_names=False)
