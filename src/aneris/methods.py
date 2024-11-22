"""
This module defines all possible functional forms of harmonization methods and
the default decision tree for choosing which method to use.
"""

from bisect import bisect

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from aneris import utils


def harmonize_factors(df, hist, harmonize_year=2015):
    """
    Calculate offset and ratio values between data and history.

    Parameters
    ----------
    df : pd.DataFrame
        model data
    hist : pd.DataFrame
        historical data
    harmonize_year : string, optional
        column name of harmonization year

    Returns
    -------
    offset : pd.Series
       offset (history - model)
    ratio : pd.Series
       ratio (history / model)
    """
    c, m = hist[harmonize_year], df[harmonize_year]
    offset = (c - m).fillna(0)
    offset.name = "offset"
    ratios = (c / m).replace(np.inf, np.nan).fillna(0)
    ratios.name = "ratio"
    return offset, ratios


def constant_offset(df, offset, harmonize_year=2015):
    """
    Calculate constant offset harmonized trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        model data
    offset : pd.DataFrame
        offset data
    harmonize_year : string, optional
        column name of harmonization year, ignored

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    numcols = utils.numcols(df)
    # just add offset to all values
    df[numcols] = df[numcols].add(offset, axis=0)
    return df


def constant_ratio(df, ratios, harmonize_year=2015):
    """
    Calculate constant ratio harmonized trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        model data
    ratio : pd.DataFrame
        ratio data
    harmonize_year : string, optional
        column name of harmonization year, ignored

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    numcols = utils.numcols(df)
    # just add offset to all values
    df[numcols] = df[numcols].multiply(ratios, axis=0)
    return df


def linear_interpolate(df, offset, final_year=2050, harmonize_year=2015):
    """
    Calculate linearly interpolated convergence harmonized trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        model data
    offset : pd.DataFrame
        offset data
    final_year : string, optional
        column name of convergence year
    harmonize_year : string, optional
        column name of harmonization year

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    x1, x2 = harmonize_year, final_year
    y1, y2 = offset + df[x1], df[x2]
    m = (y2 - y1) / (float(x2) - float(x1))
    b = y1 - m * float(x1)

    cols = [x for x in utils.numcols(df) if int(x) < int(final_year)]
    for c in cols:
        df[c] = m * float(c) + b
    return df


def reduce_offset(df, offset, final_year=2050, harmonize_year=2015):
    """
    Calculate offset convergence harmonized trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        model data
    offset : pd.DataFrame
        offset data
    final_year : string, optional
        column name of convergence year
    harmonize_year : string, optional
        column name of harmonization year

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    yi, yf = int(harmonize_year), int(final_year)
    numcols = utils.numcols(df)
    numcols_int = [int(v) for v in numcols]
    # get factors that reduce from 1 to 0; factors before base year are > 1
    f = lambda year: -(year - yi) / float(yf - yi) + 1
    factors = [f(year) if year <= yf else 0.0 for year in numcols_int]
    # add existing values to offset time series
    offsets = pd.DataFrame(
        np.outer(offset, factors), columns=numcols, index=offset.index
    )
    df[numcols] = df[numcols] + offsets
    return df


def reduce_ratio(df, ratios, final_year=2050, harmonize_year=2015):
    """
    Calculate ratio convergence harmonized trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        model data
    ratio : pd.DataFrame
        ratio data
    final_year : string, optional
        column name of convergence year
    harmonize_year : string, optional
        column name of harmonization year

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    yi, yf = int(harmonize_year), int(final_year)
    numcols = utils.numcols(df)
    numcols_int = [int(v) for v in numcols]
    # get factors that reduce from 1 to 0, but replace with 1s in years prior
    # to harmonization
    f = lambda year: -(year - yi) / float(yf - yi) + 1
    prefactors = [f(yi) for year in numcols_int if year < yi]
    postfactors = [f(year) if year <= yf else 0.0 for year in numcols_int if year >= yi]
    factors = prefactors + postfactors
    # multiply existing values by ratio time series
    ratios = (
        pd.DataFrame(np.outer(ratios - 1, factors), columns=numcols, index=ratios.index)
        + 1
    )

    df[numcols] = df[numcols] * ratios
    return df


def budget(df, df_hist, harmonize_year=2015):
    r"""
    Calculate budget harmonized trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        model data
    df_hist : pd.DataFrame
        historic data
    harmonize_year : string, optional
        column name of harmonization year

    Returns
    -------
    df_harm : pd.DataFrame
        harmonized trajectories

    Notes
    -----
    Finds an emissions trajectory consistent with a provided historical emissions
    timeseries that closely matches a modeled result, while maintaining the overall
    carbon budget.

    An optimization problem is constructed and solved by IPOPT, which minimizes the
    difference between the rate of change of the model and the harmonized model
    in each year, while
    1. preserving the carbon budget of the model, and
    2. being consistent with the historical value.

    With years :math:`y_i`, model results :math:`m_i`, harmonized results :math:`x_i`,
    historical value :math:`h_0` and a remaining carbon budget :math:`B`, the
    optimization problem can be formulated as

    .. math::

        \min_{x_i} \sum_{i \in |I - 1|}
        \big( \frac{m_{i+1} - m_i}{y_{i + 1} - y_{i}} -
                \frac{x_{i+1} - x_i}{y_{i + 1} - y_{i}} \big)^2

    s.t.

    .. math::

        \sum_{i} (y_{i + 1} - y_{i}) \big( x_i + 0.5 (x_{i+1} - x_i) \big) = B
        \quad \text{(carbon budget preservation)}

    and

    .. math::

        x_0 = h_0 \quad \text{(consistency with historical values)}
    """

    harmonize_year = int(harmonize_year)

    # df = df.set_axis(df.columns.astype(int), axis="columns")
    # df_hist = df_hist.set_axis(df_hist.columns.astype(int), axis="columns")

    data_years = df.columns
    hist_years = df_hist.columns

    years = data_years[data_years >= harmonize_year]

    if data_years[0] not in hist_years:
        hist_years = hist_years.insert(bisect(hist_years, data_years[0]), data_years[0])
        df_hist = df_hist.reindex(columns=hist_years).interpolate(
            method="slinear", axis=1
        )

    def carbon_budget(years, emissions):
        # trapezoid rule
        dyears = np.diff(years)
        demissions = np.diff(emissions)

        budget = (dyears * (np.asarray(emissions)[:-1] + demissions / 2)).sum()
        return budget

    solver = pyo.SolverFactory("ipopt")
    if solver.executable() is None:
        raise RuntimeError(
            "No executable for the solver 'ipopt' found "
            "(necessary for the budget harmonization). "
            "Install from conda-forge or add to PATH."
        )

    harmonized = []

    for region in df.index:
        model = pyo.ConcreteModel()

        """
        PARAMETERS
        """
        data_vals = df.loc[region, years]
        hist_val = df_hist.loc[region, harmonize_year]

        budget_val = carbon_budget(data_years, df.loc[region, :])

        if data_years[0] < harmonize_year:
            hist_in_overlap = df_hist.loc[region, data_years[0] : harmonize_year]
            budget_val -= carbon_budget(hist_in_overlap.index, hist_in_overlap)

        """
        VARIABLES
        """
        model.x = pyo.Var(years, initialize=0, domain=pyo.Reals)
        x = np.array(
            [model.x[y] for y in years]
        )  # keeps pyomo VarData objects, ie. modelling vars not numbers

        """
        OBJECTIVE FUNCTION
        """
        delta_years = np.diff(years)
        delta_x = np.diff(x)
        delta_m = np.diff(data_vals)

        def l2_norm():
            return pyo.quicksum((delta_m / delta_years - delta_x / delta_years) ** 2)

        model.obj = pyo.Objective(expr=l2_norm(), sense=pyo.minimize)

        """
        CONSTRAINTS
        """
        model.hist_val = pyo.Constraint(expr=model.x[harmonize_year] == hist_val)

        model.budget = pyo.Constraint(expr=carbon_budget(years, x) == budget_val)

        """
        RUN
        """
        results = solver.solve(model)

        assert (results.solver.status == pyo.SolverStatus.ok) and (
            results.solver.termination_condition == pyo.TerminationCondition.optimal
        ), (
            f"ipopt terminated budget optimization with status: "
            f"{results.solver.status}, {results.solver.termination_condition}"
        )

        harmonized.append([pyo.value(model.x[y]) for y in years])

    df_harm = pd.DataFrame(
        harmonized,
        index=df.index,
        columns=years,
    )

    return df_harm


def model_zero(df, offset, harmonize_year=2015):
    """
    Returns result of aneris.methods.constant_offset()
    """
    # current decision is to return a simple offset, this will be a straight
    # line for all time periods. previous behavior was to set df[numcols] = 0,
    # i.e., report 0 if model reports 0.
    return constant_offset(df, offset)


def hist_zero(df, *args, **kwargs):
    """
    Returns df (no change)
    """
    # TODO: should this set values to 0?
    df = df.copy()
    return df


def coeff_of_var(s):
    """
    Returns coefficient of variation of a Series.

    .. math:: c_v = \\frac{\\sigma(s^{\\prime}(t))}{\\mu(s^{\\prime}(t))}

    Parameters
    ----------
    s : pd.Series
        timeseries

    Returns
    -------
    c_v : float
        coefficient of variation
    """
    x = np.diff(s.to_numpy())
    with np.errstate(invalid="ignore"):
        return np.abs(np.std(x) / np.mean(x))


def default_method_choice(
    row,
    ratio_method="reduce_ratio_2080",
    offset_method="reduce_offset_2080",
    luc_method="reduce_offset_2150_cov",
    luc_cov_threshold=10,
):
    """
    Default decision tree as documented at.

    Refer to choice flow chart at
    https://drive.google.com/drive/folders/0B6_Oqvcg8eP9QXVKX2lFVUJiZHc
    for arguments available in row and their definition
    """
    # special cases
    if row.h == 0:
        return "hist_zero"
    if row.zero_m:
        return "model_zero"
    if np.isinf(row.f) and row.neg_m and row.pos_m:
        # model == 0 in base year, and model goes negative
        # and positive
        return "unicorn"  # this shouldn't exist!

    # model 0 in base year?
    if np.isclose(row.m, 0):
        # goes negative?
        if row.neg_m:
            return offset_method
        else:
            return "constant_offset"
    else:
        # is this co2?
        # ZN: This gas dependence isn't documented in the default
        # decision tree
        if hasattr(row, "gas") and row.gas == "CO2":
            return ratio_method
        # is cov big?
        if np.isfinite(row["cov"]) and row["cov"] > luc_cov_threshold:
            return luc_method
        else:
            # dH small?
            if row.dH < 0.5:
                return ratio_method
            else:
                # goes negative?
                if row.neg_m:
                    return "reduce_ratio_2100"
                else:
                    return "constant_ratio"


def default_methods(hist, model, base_year, method_choice=None, **kwargs):
    """
    Determine default harmonization or downscaling methods to use.

    See http://mattgidden.com/aneris/theory.html#default-decision-tree for a
    graphical description of the decision tree.

    Parameters
    ----------
    hist : pd.DataFrame
        historical data
    model : pd.DataFrame
        model data
    base_year : string, int
        harmonization year
    method_choice : function, optional
        codified decision tree, see `default_method_choice` function
    **kwargs :
        Additional parameters passed on to the choice functions.

        Harmonization functions might depend on the following method names:
        ratio_method : string
            method to use for ratio harmonization, default: reduce_ratio_2080
        offset_method : string
            method to use for offset harmonization, default: reduce_offset_2080
        luc_method : string
            method to use for high coefficient of variation, reduce_offset_2150_cov
        luc_cov_threshold : float
            cov threshold above which to use `luc_method`

        Downscaling functions require the following choices:
        intensity_method : string
            method to use for intensity convergence, default ipat_gdp_2100
        luc_method : string
            method to use for agriculture and luc emissions, default base_year_pattern

    Returns
    -------
    methods : pd.Series
       default harmonization methods
    metadata : pd.DataFrame
       metadata regarding why each method was chosen

    See also
    --------
    `default_method_choice`
    """

    y = str(base_year)
    try:
        h = hist[base_year]
        m = model[base_year]
    except KeyError:
        h = hist[y]
        m = model[y]
    dH = (h - m).abs() / h
    f = h / m
    dM = (model.max(axis=1) - model.min(axis=1)).abs() / model.max(axis=1)
    neg_m = (model < 0).any(axis=1)
    pos_m = (model > 0).any(axis=1)
    zero_m = (model == 0).all(axis=1)
    go_neg = ((model.min(axis=1) - h) < 0).any()
    cov = hist.apply(coeff_of_var, axis=1)

    df = pd.DataFrame(
        {
            "dH": dH,
            "f": f,
            "dM": dM,
            "neg_m": neg_m,
            "pos_m": pos_m,
            "zero_m": zero_m,
            "go_neg": go_neg,
            "cov": cov,
            "h": h,
            "m": m,
        }
    ).join(model.index.to_frame())

    if method_choice is None:
        method_choice = default_method_choice

    ret = df.apply(method_choice, axis=1, **kwargs)
    ret.name = "method"
    return ret, df
