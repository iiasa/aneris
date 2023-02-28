import pytest
import pandas as pd

import pandas.testing as pdt

import aneris.utils as utils
import aneris.cmip6.cmip6_utils as cmip6_utils

def test_remove_emissions_prefix():
    assert "foo" == utils.remove_emissions_prefix("foo")
    assert "foo" == utils.remove_emissions_prefix("Emissions|XXX|foo")
    assert "Emissions|bar|foo" == utils.remove_emissions_prefix("Emissions|bar|foo")
    assert "foo" == utils.remove_emissions_prefix("Emissions|bar|foo", gas="bar")



def test_no_repeat_gases():
    gases = utils.all_gases
    assert len(gases) == len(set(gases))


def test_gases():
    var_col = pd.Series(["foo|Emissions|CH4|bar", "Emissions|N2O|baz|zing"])
    exp = pd.Series(["CH4", "N2O"])
    obs = utils.gases(var_col)
    pdt.assert_series_equal(obs, exp)


def test_unit():
    var_col = pd.Series(["foo|Emissions|CH4|bar", "Emissions|N2O|baz|zing"])
    exp = pd.Series(["Mt CH4/yr", "kt N2O/yr"])
    obs = utils.units(var_col)
    pdt.assert_series_equal(obs, exp)


def combine_rows_df():
    df = pd.DataFrame(
        {
            "sector": [
                "sector1",
                "sector2",
                "sector1",
                "extra_b",
                "sector1",
            ],
            "region": ["a", "a", "b", "b", "c"],
            "2010": [1.0, 4.0, 2.0, 21, 42],
            "foo": [-1.0, -4.0, 2.0, 21, 42],
            "unit": ["Mt"] * 5,
            "gas": ["BC"] * 5,
        }
    ).set_index(utils.df_idx)
    return df

def test_isin():
    df = combine_rows_df()
    exp = pd.DataFrame(
        {
            "sector": [
                "sector1",
                "sector2",
                "sector1",
            ],
            "region": ["a", "a", "b"],
            "2010": [1.0, 4.0, 2.0],
            "foo": [-1.0, -4.0, 2.0],
            "unit": ["Mt"] * 3,
            "gas": ["BC"] * 3,
        }
    ).set_index(utils.df_idx)
    obs = exp.loc[
        utils.isin(sector=["sector1", "sector2"], region=["a", "b", "non-existent"])
    ]
    pdt.assert_frame_equal(obs, exp)

    with pytest.raises(KeyError):
        utils.isin(df, region="World", non_existing_level="foo")
