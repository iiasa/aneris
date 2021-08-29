# - can handle different units
# - can handle multiple emission scenarios
# - error if variable not in hist
# - error if region etc. not in hist
# - error if units can't be converted
# - error if harmonisation year is missing
# - can override methods for different indexes
import numpy.testing as npt
import pandas as pd
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
    out = pd.DataFrame({
        'variable': ['Emissions|BC|AFOLU', "Emissions|CO2"],
        'unit': ['Mt BC / yr', "GtC / yr"],
        'region': ["World"] * 2,
        "model": ["CEDS"] * 2,
        "scenario": ["historical"] * 2,
        '2010': [20, 10],
        '2015': [25, 15],
        '2020': [30, 20],
    })

    return out


@pytest.fixture()
def scenarios_df():
    out = pd.DataFrame({
        'variable': ['Emissions|BC|AFOLU'] * 3 + ["Emissions|CO2"] * 3,
        'unit': ['Mt BC / yr'] * 3 + ["GtC / yr"] * 3,
        'region': ["World", "a", 'World|R5.2ASIA'] * 2,
        '2010': [20, 15, 10, 10, 9, 8],
        '2015': [20, 15, 10, 10, 9, 8],
        '2020': [20, 15, 10, 10, 9, 8],
        '2030': [20, 15, 10, 10, 9, 8],
        '2050': [20, 15, 10, 10, 9, 8],
        '2100': [20, 15, 10, 10, 9, 8],
    })

    return out
