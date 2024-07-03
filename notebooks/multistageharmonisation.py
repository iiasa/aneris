# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import seaborn as sns
from concordia.report import (
    HEADING_TAGS,
    add_plotly_header,
    add_sticky_toc,
    embed_image,
)
from concordia.utils import add_totals, skipnone
from dominate import document
from dominate.tags import div
from pandas_indexing import concat, isin, ismatch
from pyomo.opt.base.solvers import OptSolver
from tqdm.auto import tqdm

from aneris import Harmonizer
# -

base_year = 2020

model = (
    pd.read_csv("data/harm_test_model.csv.gz", index_col=list(range(6)))
    .rename(columns=int)
    .droplevel(["model", "scenario"])
    .loc[isin(gas=["CO2", "CH4"])]
)
hist = (
    pd.read_csv("data/harm_test_hist.csv.gz", index_col=list(range(4)))
    .rename(columns=int)
    .pix.semijoin(model.index, how="right")
    .loc[isin(gas=["CO2", "CH4"])]
)

# +
# Simplify the problem first

agg_sectors = {
    "AFOLU": [
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Agriculture",
        "Peat Burning",
        "Deforestation and other LUC",
        "CDR Afforestation",
    ],
    "International Transport": ["International Shipping", "Aircraft"],
    "Energy and Industrial Processes": [
        "Energy Sector|Modelled",
        "Energy Sector|Non-Modelled",
        "Industrial Sector",
        "Solvents Production and Application",
        "Transportation Sector",
        "Residential Commercial Other",
    ],
    "Other (CDR and Waste)": [
        "CDR BECCS",
        "CDR DACCS",
        "CDR EW",
        "CDR Industry",
        "CDR OAE Uptake Ocean",
        "Waste",
    ],
}
model = model.pix.aggregate(sector=agg_sectors)
hist = hist.pix.aggregate(sector=agg_sectors)

# +
global_only = ["International Transport", "AFOLU"]


def filter_global_only(df, global_sectors, add_regional_totals=True):
    global_ = df.loc[isin(region="World", sector=global_sectors)]
    regional = df.loc[~isin(region="World") & ~isin(sector=global_sectors)]
    regional_agg = (
        regional.groupby(df.index.names.difference(["region"]))
        .sum()
        .pix.assign(region="World")
        if add_regional_totals
        else None
    )
    return concat(skipnone(global_, regional, regional_agg))


model = filter_global_only(model, global_only)
hist = filter_global_only(hist, global_only)


# -

# # First stage:
#
# Global gas-totals

# +
def gas_totals(df):
    return (
        df.loc[isin(region="World")]
        .groupby(df.index.names.difference(["sector"]))
        .sum()
        .pix.assign(sector="Total")
    )

model_gas_totals = gas_totals(model)
hist_gas_totals = gas_totals(hist)
# -

harmonizer = Harmonizer(
    model_gas_totals,
    hist_gas_totals,
    harm_idx=["region", "gas", "sector"],
)

harmonized_gas_totals = harmonizer.harmonize(year=base_year)
harmonized_gas_totals = harmonized_gas_totals.pix.assign(
    method=harmonizer.methods_used
    if isinstance(harmonizer.methods_used, pd.Series)
    else harmonizer.methods_used["method"]
)

harmonized_gas_totals

# # Second-stage

base_year = 2020
hist_co2 = filter_global_only(
    hist.loc[isin(gas="CO2"), base_year], global_only, add_regional_totals=False
)
model_co2 = filter_global_only(
    model.loc[isin(gas="CO2"), base_year:], global_only, add_regional_totals=False
)

solver = pyo.SolverFactory("ipopt")


def distribute_level(
    model: pd.DataFrame,
    hist: pd.Series,
    total: pd.Series,
    level: str,
    base_year: int,
    solver: str | OptSolver = "ipopt",
) -> pd.DataFrame:
    if isinstance(solver, str):
        solver = pyo.SolverFactory(solver)

    model = model.groupby(level).sum()
    hist = hist.groupby(level).sum()

    # indices
    years = model.columns
    components = model.index

    opt = pyo.ConcreteModel()
    opt.x = pyo.Var(years, components, initialize=0, domain=pyo.Reals)
    # Let's unpack those back into a dataframe for easy access
    x = (
        pd.Series(dict(opt.x.items()))
        .rename_axis(index=[years.name, level])
        .unstack(years.name)
    )

    def l2_norm():
        delta_years = years.diff()[1:]
        delta_x = x.diff(axis=1).iloc[:, 1:]
        delta_m = model.diff(axis=1).iloc[:, 1:]
        return pyo.quicksum(
            np.ravel((delta_m / delta_years - delta_x / delta_years) ** 2)
        )

    opt.obj = pyo.Objective(expr=l2_norm(), sense=pyo.minimize)

    opt.hist_val = pyo.Constraint(
        components, rule=lambda m, c: x.at[c, base_year] == hist.at[c]
    )
    opt.total_val = pyo.Constraint(
        years, rule=lambda m, y: pyo.quicksum(x[y]) == total.at[y]
    )
    results = solver.solve(opt)
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"ipopt terminated budget optimization with status: "
        f"{results.solver.status}, {results.solver.termination_condition}"
    )
    return x.map(pyo.value)


# +
model_sel = filter_global_only(
    model.loc[:, base_year:], global_only, add_regional_totals=False
)
hist_sel = filter_global_only(hist[base_year], global_only, add_regional_totals=False)

harmonized_region_totals = concat(
    distribute_level(
        model_sel.loc[isin(gas=gas)],
        hist_sel.loc[isin(gas=gas)],
        harmonized_gas_totals.loc[isin(gas=gas)].iloc[0],
        level="region",
        base_year=base_year,
        solver=solver,
    ).pix.assign(gas=gas)
    for gas in model.pix.unique("gas")
)
# -

# # 3rd stage

harmonized = concat(
    distribute_level(
        model_sel.loc[isin(gas=gas, region=region)],
        hist_sel.loc[isin(gas=gas, region=region)],
        total=harmonized_region_totals.loc[isin(gas=gas, region=region)].iloc[0],
        level="sector",
        base_year=base_year,
        solver=solver,
    ).pix.assign(gas=gas, region=region, order=["gas", "region", "sector"])
    for gas, region in model.pix.unique(["gas", "region"])
).sort_index()

harmonized = (
    harmonized.pix.semijoin(model.index) # hack the unit definitions back :)
)

# # Plotting

# +
model = add_totals(model)
hist = add_totals(hist)

# TODO possibly some double counting happening here, should be debugged
harmonized = add_totals(harmonized)


# -

def plot_harm(sel, scenario=None, levels=["gas", "sector", "region"], useplotly=False):
    model_sel = sel if scenario is None else sel & ismatch(scenario=scenario)
    h = harmonized.loc[model_sel]

    data = concat(
        dict(
            History=hist.loc[sel],
            Unharmonized=model.loc[model_sel],
            Harmonized=h,
        ),
        keys="pathway",
    )

    non_uniques = [lvl for lvl in levels if len(h.pix.unique(lvl)) > 1]
    if not non_uniques:
        non_uniques = ["region"]
        data = data.pix.semijoin(h.pix.unique(levels), how="right")

    (non_unique,) = non_uniques

    if useplotly:
        g = px.line(
            data.pix.to_tidy(),
            x="year",
            y="value",
            color="pathway",
            style="version",
            facet_col=non_unique,
            facet_col_wrap=4,
            labels=dict(value=data.pix.unique("unit").item(), pathway="Trajectory"),
        )
        g.update_yaxes(matches=None)

    num_facets = len(data.pix.unique(non_unique))
    multirow_args = dict(col_wrap=4, height=2, aspect=1.5) if num_facets > 1 else dict()
    g = sns.relplot(
        data.pix.to_tidy(),
        kind="line",
        x="year",
        y="value",
        col=non_unique,
        hue="pathway",
        facet_kws=dict(sharey=False),
        legend=True,
        **multirow_args
    ).set(ylabel=data.pix.unique("unit").item())
    for label, ax in g.axes_dict.items():
        ax.set_title(f"{non_unique} = {label}", fontsize=9)
    return g


# +
def what_changed(next, prev):
    length = len(next)
    if prev is None:
        return range(length)
    for i in range(length):
        if prev[i] != next[i]:
            return range(i, length)

def make_doc(order, compact=False, useplotly=False):
    index = harmonized.index.pix.unique(order).sort_values()
    doc = document(title="Harmonization results")

    main = doc.add(div())
    prev_idx = None
    for idx in tqdm(index):
        main.add([HEADING_TAGS[i](idx[i]) for i in what_changed(idx, prev_idx)])

        try:
            ax = plot_harm(
                isin(**dict(zip(index.names, idx)), ignore_missing_levels=True),
                useplotly=useplotly,
            )
        except ValueError:
            print(
                f"During plot_harm(isin(**{dict(zip(index.names, idx))}, ignore_missing_levels=True))"
            )
            raise
        main.add(embed_image(ax, close=True))

        prev_idx = idx

    add_sticky_toc(doc, max_level=2, compact=compact)
    if useplotly:
        add_plotly_header(doc)
    return doc


# -

out_path = Path("results")
out_path.mkdir(exist_ok=True)


# +
def make_scenario_facets(useplotly=False):
    suffix = "-plotly" if useplotly else ""
    fn = out_path / f"harmonization-facet{suffix}.html"

    #lock = FileLock(out_path / ".lock")
    doc = make_doc(order=["gas", "sector"], useplotly=useplotly)

    #with lock:
    with open(fn, "w", encoding="utf-8") as f:
        print(doc, file=f)
    return fn

make_scenario_facets()
# -

# !open results/harmonization-facet.html
