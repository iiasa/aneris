import pytest
import pandas as pd

import pandas.testing as pdt

import aneris.utils as utils
import aneris.cmip6.cmip6_utils as cutils


from test_utils import combine_rows_df


def test_region_agg_funky_name():
    df = (
        pd.DataFrame(
            {
                "sector": ["foo", "foo"],
                "region": ["a", "b"],
                "2010": [1.0, 4.0],
                "unit": ["Mt"] * 2,
                "gas": ["BC"] * 2,
            }
        )
        .set_index(utils.df_idx)
        .sort_index()
    )
    mapping = pd.DataFrame([["fOO_Bar", "a"], ["fOO_Bar", "b"]], columns=["x", "y"])
    exp = (
        pd.DataFrame(
            {
                "sector": ["foo"],
                "region": ["fOO_Bar"],
                "2010": [5.0],
                "unit": ["Mt"],
                "gas": ["BC"],
            }
        )
        .set_index(utils.df_idx)
        .sort_index()
    )
    obs = cutils.agg_regions(df, rfrom="y", rto="x", mapping=mapping)
    pdt.assert_frame_equal(obs, exp)


def test_formatter_to_std():
    df = pd.DataFrame(
        {
            "Variable": [
                "CEDS+|9+ Sectors|Emissions|BC|foo|Unharmonized",
                "Emissions|BC|bar|baz",
            ],
            "Region": ["a", "b"],
            "2010": [5.0, 2.0],
            "2020": [-1.0, 3.0],
            "Unit": ["Mt foo/yr"] * 2,
            "Model": ["foo"] * 2,
            "Scenario": ["foo"] * 2,
        }
    )

    fmt = cutils.FormatTranslator(df.copy())
    obs = fmt.to_std()
    exp = pd.DataFrame(
        {
            "sector": [
                "CEDS+|9+ Sectors|foo|Unharmonized",
                "bar|baz",
            ],
            "region": ["a", "b"],
            "2010": [5000.0, 2000.0],
            "2020": [-1000.0, 3000.0],
            "unit": ["kt"] * 2,
            "gas": ["BC"] * 2,
        }
    )
    pdt.assert_frame_equal(obs.set_index(cutils.df_idx), exp.set_index(cutils.df_idx))


def test_formatter_to_template():
    df = pd.DataFrame(
        {
            "Variable": [
                "CEDS+|9+ Sectors|Emissions|BC|foo|Unharmonized",
                "CEDS+|9+ Sectors|Emissions|BC|bar|Unharmonized",
            ],
            "Region": ["a", "b"],
            "2010": [5.0, 2.0],
            "2020": [-1.0, 3.0],
            "Unit": ["Mt BC/yr"] * 2,
            "Model": ["foo"] * 2,
            "Scenario": ["foo"] * 2,
        }
    ).set_index(cutils.iamc_idx)
    fmt = cutils.FormatTranslator(df, prefix="CEDS+|9+ Sectors", suffix="Unharmonized")
    fmt.to_std()
    obs = fmt.to_template()
    exp = df.reindex(columns=obs.columns)
    pdt.assert_frame_equal(obs, exp)


def test_combine_rows_default():
    df = combine_rows_df()
    exp = pd.DataFrame(
        {
            "sector": [
                "sector1",
                "sector2",
                "extra_b",
                "sector1",
            ],
            "region": ["a", "a", "a", "c"],
            "2010": [3.0, 4.0, 21, 42],
            "foo": [1.0, -4.0, 21, 42],
            "unit": ["Mt"] * 4,
            "gas": ["BC"] * 4,
        }
    ).set_index(cutils.df_idx)
    obs = cutils.combine_rows(df, "region", "a", ["b"])

    exp = exp.reindex(columns=obs.columns)
    clean = lambda df: df.sort_index().reset_index()
    pdt.assert_frame_equal(clean(obs), clean(exp))


def test_combine_rows_dropothers():
    df = combine_rows_df()
    exp = pd.DataFrame(
        {
            "sector": [
                "sector1",
                "sector2",
                "extra_b",
                "sector1",
                "extra_b",
                "sector1",
            ],
            "region": ["a", "a", "a", "b", "b", "c"],
            "2010": [3.0, 4.0, 21, 2.0, 21, 42],
            "foo": [1.0, -4.0, 21, 2.0, 21, 42],
            "unit": ["Mt"] * 6,
            "gas": ["BC"] * 6,
        }
    ).set_index(cutils.df_idx)
    obs = cutils.combine_rows(df, "region", "a", ["b"], dropothers=False)

    exp = exp.reindex(columns=obs.columns)
    clean = lambda df: df.sort_index().reset_index()
    pdt.assert_frame_equal(clean(obs), clean(exp))


def test_combine_rows_sumall():
    df = combine_rows_df()
    exp = pd.DataFrame(
        {
            "sector": [
                "sector1",
                "extra_b",
                "sector1",
            ],
            "region": ["a", "a", "c"],
            "2010": [2.0, 21, 42],
            "foo": [2.0, 21, 42],
            "unit": ["Mt"] * 3,
            "gas": ["BC"] * 3,
        }
    ).set_index(cutils.df_idx)
    obs = cutils.combine_rows(df, "region", "a", ["b"], sumall=False)

    exp = exp.reindex(columns=obs.columns)
    clean = lambda df: df.sort_index().reset_index()
    pdt.assert_frame_equal(clean(obs), clean(exp))