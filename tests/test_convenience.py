# - default decision tree applied properly
# - error if variable not in hist
# - error if region etc. not in hist
# - error if units can't be converted
# - error if harmonisation year is missing
# - can override methods for different indexes
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from aneris.convenience import harmonise_all


@pytest.mark.parametrize("method,exp_res", (
    (
        "constant_ratio",
        {
            2010: 10 * 1.1,
            2030: 5 * 1.1,
            2050: 3 * 1.1,
            2100: 1 * 1.1,
        }
    ),
    (
        "reduce_ratio_2050",
        {
            2010: 11,
            2030: 5 * 1.05,
            2050: 3,
            2100: 1,
        }
    ),
    (
        "reduce_ratio_2030",
        {
            2010: 11,
            2030: 5,
            2050: 3,
            2100: 1,
        }
    ),
    (
        "reduce_ratio_2150",
        {
            2010: 11,
            2030: 5 * (1 + 0.1 * (140 - 20) / 140),
            2050: 3 * (1 + 0.1 * (140 - 40) / 140),
            2100: 1 * (1 + 0.1 * (140 - 90) / 140),
        }
    ),
    (
        "constant_offset",
        {
            2010: 10 + 1,
            2030: 5 + 1,
            2050: 3 + 1,
            2100: 1 + 1,
        }
    ),
    (
        "reduce_offset_2050",
        {
            2010: 11,
            2030: 5 + 0.5,
            2050: 3,
            2100: 1,
        }
    ),
    (
        "reduce_offset_2030",
        {
            2010: 11,
            2030: 5,
            2050: 3,
            2100: 1,
        }
    ),
    (
        "reduce_offset_2150",
        {
            2010: 11,
            2030: 5 + 1 * (140 - 20) / 140,
            2050: 3 + 1 * (140 - 40) / 140,
            2100: 1 + 1 * (140 - 90) / 140,
        }
    ),
    (
        "model_zero",
        {
            2010: 10 + 1,
            2030: 5 + 1,
            2050: 3 + 1,
            2100: 1 + 1,
        }
    ),
    (
        "hist_zero",
        {
            2010: 10,
            2030: 5,
            2050: 3,
            2100: 1,
        }
    ),
))
def test_different_unit_handling(method, exp_res):
    idx = ["variable", "unit", "region", "model", "scenario"]

    hist = pd.DataFrame({
        'variable': ["Emissions|CO2"],
        'unit': ["MtC / yr"],
        'region': ["World"],
        "model": ["CEDS"],
        "scenario": ["historical"],
        2010: [11000],
    }).set_index(idx)

    scenario = pd.DataFrame({
        'variable': ["Emissions|CO2"],
        'unit': ["GtC / yr"],
        'region': ["World"],
        "model": ["IAM"],
        "scenario": ["abc"],
        2010: [10],
        2030: [5],
        2050: [3],
        2100: [1],
    }).set_index(idx)

    overrides = [
        {"variable": "Emissions|CO2", "method": method}
    ]
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

    hist = pd.DataFrame({
        'variable': ["Emissions|CO2", "Emissions|CH4"],
        'unit': ["MtCO2 / yr", "MtCH4 / yr"],
        'region': ["World"] * 2,
        "model": ["CEDS"] * 2,
        "scenario": ["historical"] * 2,
        2010: [11000 * 44 / 12, 200],
        2015: [12000 * 44 / 12, 250],
        2020: [13000 * 44 / 12, 300],
    }).set_index(idx)

    return hist


@pytest.fixture()
def scenarios_df():
    idx = ["variable", "unit", "region", "model", "scenario"]

    scenario = pd.DataFrame({
        'variable': ["Emissions|CO2", "Emissions|CH4"],
        'unit': ["GtC / yr", "GtCH4 / yr"],
        'region': ["World"] * 2,
        "model": ["IAM"] * 2,
        "scenario": ["abc"] * 2,
        2010: [10, 0.1],
        2015: [11, 0.15],
        2020: [5, 0.25],
        2030: [5, 0.1],
        2050: [3, 0.05],
        2100: [1, 0.03],
    }).set_index(idx)

    return scenario


@pytest.mark.parametrize("extra_col", (False, "mip_era"))
@pytest.mark.parametrize("harmonisation_year,scales", (
    (2010, [1.1, 2]),
    (2015, [12 / 11, 25 / 15]),
))
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

    overrides = [
        {"method": "constant_ratio"}
    ]
    overrides = pd.DataFrame(overrides)

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        harmonisation_year=harmonisation_year,
        overrides=overrides,
    )

    pdt.assert_frame_equal(res, exp)


@pytest.mark.parametrize("harmonisation_year,offset", (
    (2010, [1, 0.1]),
    (2015, [1, 0.1]),
    (2020, [8, 0.05]),
))
def test_different_unit_handling_multiple_timeseries_constant_offset(
    hist_df,
    scenarios_df,
    harmonisation_year,
    offset,
):
    exp = scenarios_df.add(offset, axis=0)

    overrides = [
        {"method": "constant_offset"}
    ]
    overrides = pd.DataFrame(overrides)

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        harmonisation_year=harmonisation_year,
        overrides=overrides,
    )

    pdt.assert_frame_equal(res, exp)
