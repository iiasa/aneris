import re

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from aneris.convenience import harmonise_all
from aneris.errors import (
    AmbiguousHarmonisationMethod,
    MissingHarmonisationYear,
    MissingHistoricalError,
)


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

    overrides = [{"variable": "Emissions|CO2", "region": "World", "method": method}]
    overrides = pd.DataFrame(overrides).set_index(["variable", "region"])["method"]

    res = harmonise_all(
        scenarios=scenario,
        history=hist,
        year=2010,
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
    "year,scales",
    (
        (2010, [1.1, 2]),
        (2015, [12 / 11, 25 / 15]),
    ),
)
def test_different_unit_handling_multiple_timeseries_constant_ratio(
    hist_df,
    scenarios_df,
    extra_col,
    year,
    scales,
):
    if extra_col:
        scenarios_df[extra_col] = "test"
        scenarios_df = scenarios_df.set_index(extra_col, append=True)

    exp = scenarios_df.multiply(scales, axis=0)
    # new requirement - we won't provide data before harmonization year
    exp = exp[[c for c in exp.columns if c >= year]]

    overrides = [{"region": "World", "method": "constant_ratio"}]
    overrides = pd.DataFrame(overrides).set_index(["region"])["method"]

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        year=year,
        overrides=overrides,
    )
    pdt.assert_frame_equal(res, exp)


@pytest.mark.parametrize(
    "year,offset",
    (
        (2010, [1, 0.1]),
        (2015, [1, 0.1]),
        (2020, [8, 0.05]),
    ),
)
def test_different_unit_handling_multiple_timeseries_constant_offset(
    hist_df,
    scenarios_df,
    year,
    offset,
):
    exp = scenarios_df.add(offset, axis=0)
    # new requirement - we won't provide data before harmonization year
    exp = exp[[c for c in exp.columns if c >= year]]

    overrides = [{"method": "constant_offset"}]
    overrides = pd.DataFrame(overrides)

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        year=year,
        overrides=overrides,
    )

    pdt.assert_frame_equal(res, exp)


def test_different_unit_handling_multiple_timeseries_overrides(
    hist_df,
    scenarios_df,
):
    year = 2015

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
                    sf = 1 + ((harm_year_ratio - 1) * (2050 - c) / (2050 - year))

                exp.loc[r, c] *= sf
            else:
                harm_year_offset = 0.1

                if c >= 2030:
                    of = 0
                else:
                    of = harm_year_offset * (2030 - c) / (2030 - year)

                exp.loc[r, c] += of
    # new requirement - we won't provide data before harmonization year
    exp = exp[[c for c in exp.columns if c >= year]]

    overrides = [
        {"variable": "Emissions|CO2", "method": "reduce_ratio_2050"},
        {"variable": "Emissions|CH4", "method": "reduce_offset_2030"},
    ]
    overrides = pd.DataFrame(overrides).set_index("variable")["method"]

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        year=year,
        overrides=overrides,
    )
    pdt.assert_frame_equal(res, exp, check_like=True)


def test_raise_if_variable_not_in_hist(hist_df, scenarios_df):
    hist_df = hist_df[~hist_df.index.get_level_values("variable").str.endswith("CO2")]

    with pytest.raises(MissingHistoricalError):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            year=2010,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_raise_if_incompatible_unit(hist_df, scenarios_df):
    scenarios_df = scenarios_df.reset_index("unit")
    scenarios_df["unit"] = "Mt CO2 / yr"
    scenarios_df = scenarios_df.set_index("unit", append=True)

    error_msg = re.escape(
        "Cannot convert from 'megatCH4 / yr' ([mass] * [methane] / [time]) to 'megametric_ton * CO2 / yr' ([mass] * [carbon] / [time])"
    )
    with pytest.raises(pint.errors.DimensionalityError, match=error_msg):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            year=2010,
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
            year=2010,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_raise_if_year_missing(hist_df, scenarios_df):
    hist_df = hist_df.drop(2015, axis="columns")

    error_msg = re.escape("No historical data in harmonization year")
    with pytest.raises(MissingHarmonisationYear, match=error_msg):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            year=2015,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_raise_if_year_nan(hist_df, scenarios_df):
    hist_df.loc[
        hist_df.index.get_level_values("variable").str.endswith("CO2"), 2015
    ] = np.nan

    with pytest.raises(MissingHistoricalError):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            year=2015,
            overrides=pd.DataFrame([{"method": "constant_ratio"}]),
        )


def test_override_multi_level(hist_df, scenarios_df):
    asia_hist = hist_df * 0.7
    asia_hist.index = asia_hist.index.set_levels(["World|R5.2ASIA"], level="region")

    hist_df = pd.concat([hist_df, asia_hist])

    asia = scenarios_df.copy()
    asia.index = asia.index.set_levels(["World|R5.2ASIA"], level="region")

    model_2 = scenarios_df.copy()
    model_2.index = model_2.index.set_levels(["FaNCY"], level="model")

    scenario_2 = scenarios_df.copy()
    scenario_2.index = scenario_2.index.set_levels(["EMF33 quick"], level="scenario")

    scenarios_df = pd.concat([scenarios_df, asia, model_2, scenario_2])

    overrides = pd.DataFrame(
        [
            {
                "variable": "Emissions|CO2",
                "region": "World",
                "model": "IAM",
                "scenario": "abc",
                "method": "constant_ratio",
            },
            {
                "variable": "Emissions|CH4",
                "region": "World",
                "model": "IAM",
                "scenario": "abc",
                "method": "constant_offset",
            },
            {
                "variable": "Emissions|CO2",
                "region": "World|R5.2ASIA",
                "model": "IAM",
                "scenario": "abc",
                "method": "reduce_ratio_2030",
            },
            {
                "variable": "Emissions|CH4",
                "region": "World|R5.2ASIA",
                "model": "IAM",
                "scenario": "abc",
                "method": "reduce_ratio_2050",
            },
            {
                "variable": "Emissions|CO2",
                "region": "World",
                "model": "FaNCY",
                "scenario": "abc",
                "method": "reduce_ratio_2070",
            },
            {
                "variable": "Emissions|CH4",
                "region": "World",
                "model": "FaNCY",
                "scenario": "abc",
                "method": "reduce_ratio_2090",
            },
            {
                "variable": "Emissions|CO2",
                "region": "World",
                "model": "IAM",
                "scenario": "EMF33 quick",
                "method": "reduce_offset_2050",
            },
            {
                "variable": "Emissions|CH4",
                "region": "World",
                "model": "IAM",
                "scenario": "EMF33 quick",
                "method": "reduce_offset_2070",
            },
        ]
    ).set_index(["model", "scenario", "region", "variable"])["method"]

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        year=2015,
        overrides=overrides,
    )

    co2_rows = res.index.get_level_values("variable") == "Emissions|CO2"
    world_rows = res.index.get_level_values("region") == "World"
    fancy_rows = res.index.get_level_values("model") == "FaNCY"
    emf33_rows = res.index.get_level_values("scenario") == "EMF33 quick"

    atol = 1e-4
    pick_rows = co2_rows & world_rows & ~fancy_rows & ~emf33_rows
    scenarios_df = scenarios_df[[c for c in scenarios_df.columns if c >= 2015]]
    npt.assert_allclose(
        res.loc[pick_rows, :],
        12 / 11 * scenarios_df.loc[pick_rows, :],
        atol=atol,
    )
    npt.assert_allclose(
        res.loc[~co2_rows & world_rows & ~fancy_rows & ~emf33_rows, :],
        0.1 + scenarios_df.loc[~co2_rows & world_rows & ~fancy_rows & ~emf33_rows, :],
        atol=atol,
    )

    npt.assert_allclose(
        res.loc[co2_rows & ~world_rows & ~fancy_rows & ~emf33_rows, :].squeeze(),
        [8.4, 4.21212121, 5, 3, 1],
        atol=atol,
    )
    npt.assert_allclose(
        res.loc[~co2_rows & ~world_rows & ~fancy_rows & ~emf33_rows, :].squeeze(),
        [0.175, 0.285714, 0.109524, 0.05, 0.03],
        atol=atol,
    )

    npt.assert_allclose(
        res.loc[co2_rows & world_rows & fancy_rows & ~emf33_rows, :].squeeze(),
        [12, 5.413233, 5.330579, 3.099174, 1],
        atol=atol,
    )
    npt.assert_allclose(
        res.loc[~co2_rows & world_rows & fancy_rows & ~emf33_rows, :].squeeze(),
        [0.25, 0.405555, 0.15333, 0.067777, 0.03],
        atol=atol,
    )

    npt.assert_allclose(
        res.loc[co2_rows & world_rows & ~fancy_rows & emf33_rows, :].squeeze(),
        [12, 5.857143, 5.571429, 3, 1],
        atol=atol,
    )
    npt.assert_allclose(
        res.loc[~co2_rows & world_rows & ~fancy_rows & emf33_rows, :].squeeze(),
        [0.25, 0.340909, 0.172727, 0.086364, 0.03],
        atol=atol,
    )


@pytest.mark.parametrize(
    "overrides",
    (
        pd.DataFrame(
            [
                {"region": "World", "method": "constant_ratio"},
                {"region": "World", "method": "constant_offset"},
            ]
        ).set_index("region")["method"],
        pd.DataFrame(
            [
                {
                    "region": "World",
                    "variable": "Emissions|CH4",
                    "method": "constant_ratio",
                },
                {"region": "World", "method": "constant_offset"},
            ]
        ),
        pd.DataFrame(
            [
                {"variable": "Emissions|CH4", "method": "constant_ratio"},
                {"variable": "Emissions|CH4", "method": "reduce_offset_2030"},
            ]
        ),
        pd.DataFrame(
            [
                {"variable": "Emissions|CH4", "method": "constant_ratio"},
                {
                    "variable": "Emissions|CH4",
                    "model": "IAM",
                    "method": "reduce_offset_2030",
                },
            ]
        ),
    ),
)
def test_multiple_matching_overrides(hist_df, scenarios_df, overrides):
    with pytest.raises(
        AmbiguousHarmonisationMethod,
    ):
        harmonise_all(
            scenarios=scenarios_df,
            history=hist_df,
            year=2015,
            overrides=overrides,
        )


def test_defaults(hist_df, scenarios_df):
    co2_afolu = scenarios_df[
        scenarios_df.index.get_level_values("variable") == "Emissions|CO2"
    ].copy()
    co2_afolu = co2_afolu.reset_index()
    co2_afolu["variable"] = "Emissions|CO2|AFOLU"
    co2_afolu = co2_afolu.set_index(scenarios_df.index.names)
    co2_afolu.iloc[:, :] = [2, 0.5, -1, -1.5, -2, -3]

    bc_afolu = scenarios_df[
        scenarios_df.index.get_level_values("variable") == "Emissions|CO2"
    ].copy()
    bc_afolu = bc_afolu.reset_index()
    bc_afolu["variable"] = "Emissions|BC|AFOLU"
    bc_afolu["unit"] = "Mt BC / yr"
    bc_afolu = bc_afolu.set_index(scenarios_df.index.names)
    bc_afolu.iloc[:, :] = [30, 33, 40, 42, 36, 24]

    scenarios_df = pd.concat([scenarios_df, co2_afolu, bc_afolu])

    co2_afolu_hist = hist_df[
        hist_df.index.get_level_values("variable") == "Emissions|CO2"
    ].copy()
    co2_afolu_hist = co2_afolu_hist.reset_index()
    co2_afolu_hist["variable"] = "Emissions|CO2|AFOLU"
    co2_afolu_hist = co2_afolu_hist.set_index(hist_df.index.names)
    co2_afolu_hist.iloc[:, :] = [
        1.5 * 44000 / 12,
        1.6 * 44000 / 12,
        1.7 * 44000 / 12,
    ]

    bc_afolu_hist = hist_df[
        hist_df.index.get_level_values("variable") == "Emissions|CO2"
    ].copy()
    bc_afolu_hist = bc_afolu_hist.reset_index()
    bc_afolu_hist["variable"] = "Emissions|BC|AFOLU"
    bc_afolu_hist["unit"] = "Gt BC / yr"
    bc_afolu_hist = bc_afolu_hist.set_index(hist_df.index.names)
    bc_afolu_hist.iloc[:, :] = [20, 35, 28]

    hist_df = pd.concat([hist_df, co2_afolu_hist, bc_afolu_hist])

    res = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        year=2015,
    )

    exp = harmonise_all(
        scenarios=scenarios_df,
        history=hist_df,
        year=2015,
        overrides=pd.DataFrame(
            [
                {"variable": "Emissions|CO2", "method": "reduce_ratio_2080"},
                {"variable": "Emissions|CH4", "method": "reduce_ratio_2080"},
                {"variable": "Emissions|CO2|AFOLU", "method": "reduce_ratio_2100"},
                {"variable": "Emissions|BC|AFOLU", "method": "constant_ratio"},
            ]
        ),
    )

    pdt.assert_frame_equal(res, exp, check_like=True)
