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


def test_different_unit_handling():
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
        {"variable": "Emissions|CO2", "method": "reduce_ratio_2050"}
    ]
    overrides = pd.DataFrame(overrides)

    res = harmonise_all(
        scenarios=scenario,
        history=hist,
        harmonisation_year=2010,
        overrides=overrides,
    )

    npt.assert_allclose(res[2010], 11)
    npt.assert_allclose(res[2030], 5 * 1.05)
    npt.assert_allclose(res[2050], 3)
    npt.assert_allclose(res[2100], 1)


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
