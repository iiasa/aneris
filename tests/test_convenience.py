# Tests to write:
# - default decision tree applied properly
# - can override methods for different indexes (specified at different levels)
import re

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from aneris.convenience import harmonise_all
from aneris.errors import MissingHarmonisationYear, MissingHistoricalError

pytest.importorskip("pint")
import pint.errors


@pytest.mark.parametrize(
    "method,exp_res",
    (
        (
            "constant_ratio",
            {
                2010: 10 * 1.1,
                2030: 5 * 1.1,
                2050: 3 * 1.1,
                2100: 1 * 1.1,
            },
        ),
        (
            "reduce_ratio_2050",
            {
                2010: 11,
                2030: 5 * 1.05,
                2050: 3,
                2100: 1,
            },
        ),
        (
            "reduce_ratio_2030",
            {
                2010: 11,
                2030: 5,
                2050: 3,
                2100: 1,
            },
        ),
        (
            "reduce_ratio_2150",
            {
                2010: 11,
                2030: 5 * (1 + 0.1 * (140 - 20) / 140),
                2050: 3 * (1 + 0.1 * (140 - 40) / 140),
                2100: 1 * (1 + 0.1 * (140 - 90) / 140),
            },
        ),
        (
            "constant_offset",
            {
                2010: 10 + 1,
                2030: 5 + 1,
                2050: 3 + 1,
                2100: 1 + 1,
            },
        ),
        (
            "reduce_offset_2050",
            {
                2010: 11,
                2030: 5 + 0.5,
                2050: 3,
                2100: 1,
            },
        ),
        (
            "reduce_offset_2030",
            {
                2010: 11,
                2030: 5,
                2050: 3,
                2100: 1,
            },
        ),
        (
            "reduce_offset_2150",
            {
                2010: 11,
                2030: 5 + 1 * (140 - 20) / 140,
                2050: 3 + 1 * (140 - 40) / 140,
                2100: 1 + 1 * (140 - 90) / 140,
            },
        ),
        (
            "model_zero",
            {
                2010: 10 + 1,
                2030: 5 + 1,
                2050: 3 + 1,
                2100: 1 + 1,
            },
        ),
        (
            "hist_zero",
            {
                2010: 10,
                2030: 5,
                2050: 3,
                2100: 1,
            },
        ),
    ),
)
def test_different_unit_handling(method, exp_res):
    idx = ["variable", "unit", "region", "model", "scenario"]

    hist = pd.DataFrame(
        {
            "variable": ["Emissions|CO2"],
            "unit": ["MtC / yr"],
            "region": ["World"],
            "model": ["CEDS"],
            "scenario": ["historical"],
            2010: [11000],
        }
    ).set_index(idx)

    scenario = pd.DataFrame(
        {
            "variable": ["Emissions|CO2"],
            "unit": ["GtC / yr"],
            "region": ["World"],
            "model": ["IAM"],
            "scenario": ["abc"],
            2010: [10],
            2030: [5],
            2050: [3],
            2100: [1],
        }
    ).set_index(idx)

    overrides = [{"variable": "Emissions|CO2", "method": method}]
    overrides = pd.DataFrame(overrides)

    res = harmonise_all(
        scenarios=scenario,
        history=hist,
        harmonisation_year=2010,
        overrides=overrides,
    )

    for year, val in exp_res.items():
        npt.assert_allclose(res[year], val)


@pytest.fixture()
def hist_df():
    idx = ["variable", "unit", "region", "model", "scenario"]

    hist = pd.DataFrame(
        {
            "variable": ["Emissions|CO2", "Emissions|CH4"],
            "unit": ["MtCO2 / yr", "MtCH4 / yr"],
            "region": ["World"] * 2,
            "model": ["CEDS"] * 2,
            "scenario": ["historical"] * 2,
            2010: [11000 * 44 / 12, 200],
            2015: [12000 * 44 / 12, 250],
            2020: [13000 * 44 / 12, 300],
        }
    ).set_index(idx)

    return hist


@pytest.fixture()
def scenarios_df():
    idx = ["variable", "unit", "region", "model", "scenario"]

    scenario = pd.DataFrame(
        {
            "variable": ["Emissions|CO2", "Emissions|CH4"],
            "unit": ["GtC / yr", "GtCH4 / yr"],
            "region": ["World"] * 2,
            "model": ["IAM"] * 2,
            "scenario": ["abc"] * 2,
            2010: [10, 0.1],
            2015: [11, 0.15],
            2020: [5, 0.25],
            2030: [5, 0.1],
            2050: [3, 0.05],
            2100: [1, 0.03],
        }
    ).set_index(idx)

    return scenario


@pytest.mark.parametrize("extra_col", (False, "mip_era"))
@pytest.mark.parametrize(
    "harmonisation_year,scales",
    (
        (2010, [1.1, 2]),
        (2015, [12 / 11, 25 / 15]),
    ),
)
def test_different_unit_handling_multiple_timeseries_constant_ratio(
    hist_df,
    scenarios_df,
    extra_col,
    harmonisation_year,
    scales,
):
    if extra_col:
        scenarios_df[extra_col] = "test"
        scenarios_df = scenarios_df.set_index(extra_col, append=True)

    exp = scenarios_df.multiply(scales, axis=0)

    overrides = [{"method": "constant_ratio"}]
    overrides = pd.DataFrame(overrides)

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        harmonisation_year=harmonisation_year,
        overrides=overrides,
    )

    pdt.assert_frame_equal(res, exp)


@pytest.mark.parametrize(
    "harmonisation_year,offset",
    (
        (2010, [1, 0.1]),
        (2015, [1, 0.1]),
        (2020, [8, 0.05]),
    ),
)
def test_different_unit_handling_multiple_timeseries_constant_offset(
    hist_df,
    scenarios_df,
    harmonisation_year,
    offset,
):
    exp = scenarios_df.add(offset, axis=0)

    overrides = [{"method": "constant_offset"}]
    overrides = pd.DataFrame(overrides)

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        harmonisation_year=harmonisation_year,
        overrides=overrides,
    )

    pdt.assert_frame_equal(res, exp)


def test_different_unit_handling_multiple_timeseries_overrides(
    hist_df,
    scenarios_df,
):
    harmonisation_year = 2015

    exp = scenarios_df.sort_index()
    for r in exp.index:
        for c in exp:
            if "CO2" in r[0]:
                harm_year_ratio = 12 / 11

                if c >= 2050:
                    sf = 1
                elif c <= 2015:
                    # this custom pre-harmonisation year logic doesn't apply to
                    # offsets which seems surprising
                    sf = harm_year_ratio
                else:
                    sf = 1 + (
                        (harm_year_ratio - 1) * (2050 - c) / (2050 - harmonisation_year)
                    )

                exp.loc[r, c] *= sf
            else:
                harm_year_offset = 0.1

                if c >= 2030:
                    of = 0
                else:
                    of = harm_year_offset * (2030 - c) / (2030 - harmonisation_year)

                exp.loc[r, c] += of

    overrides = [
        {"variable": "Emissions|CO2", "method": "reduce_ratio_2050"},
        {"variable": "Emissions|CH4", "method": "reduce_offset_2030"},
    ]
    overrides = pd.DataFrame(overrides)

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        harmonisation_year=harmonisation_year,
        overrides=overrides,
    )

    pdt.assert_frame_equal(res, exp, check_like=True)


def test_raise_if_variable_not_in_hist(hist_df, scenarios_df):
    hist_df = hist_df[~hist_df.index.get_level_values("variable").str.endswith("CO2")]

    error_msg = re.escape("No historical data for `World` `Emissions|CO2`")
    with pytest.raises(MissingHistoricalError, match=error_msg):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            harmonisation_year=2010,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_raise_if_region_not_in_hist(hist_df, scenarios_df):
    hist_df = hist_df[~hist_df.index.get_level_values("region").str.startswith("World")]

    error_msg = re.escape("No historical data for `World` `Emissions|CH4`")
    with pytest.raises(MissingHistoricalError, match=error_msg):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            harmonisation_year=2010,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_raise_if_incompatible_unit(hist_df, scenarios_df):
    scenarios_df = scenarios_df.reset_index("unit")
    scenarios_df["unit"] = "Mt CO2 / yr"
    scenarios_df = scenarios_df.set_index("unit", append=True)

    error_msg = re.escape(
        "Cannot convert from 'megatCH4 / a' ([mass] * [methane] / [time]) to "
        "'CO2 * megametric_ton / a' ([carbon] * [mass] / [time])"
    )
    with pytest.raises(pint.errors.DimensionalityError, match=error_msg):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            harmonisation_year=2010,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_raise_if_undefined_unit(hist_df, scenarios_df):
    scenarios_df = scenarios_df.reset_index("unit")
    scenarios_df["unit"] = "Mt CO2eq / yr"
    scenarios_df = scenarios_df.set_index("unit", append=True)

    with pytest.raises(pint.errors.UndefinedUnitError):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            harmonisation_year=2010,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_raise_if_harmonisation_year_missing(hist_df, scenarios_df):
    hist_df = hist_df.drop(2015, axis="columns")

    error_msg = re.escape(
        "No historical data for year 2015 for `World` `Emissions|CH4`"
    )
    with pytest.raises(MissingHarmonisationYear, match=error_msg):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            harmonisation_year=2015,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_raise_if_harmonisation_year_nan(hist_df, scenarios_df):
    hist_df.loc[
        hist_df.index.get_level_values("variable").str.endswith("CO2"), 2015
    ] = np.nan

    error_msg = re.escape(
        "Historical data is null for year 2015 for `World` `Emissions|CO2`"
    )
    with pytest.raises(MissingHarmonisationYear, match=error_msg):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            harmonisation_year=2015,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )
