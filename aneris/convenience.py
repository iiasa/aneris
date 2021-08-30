from openscm_units import unit_registry

from .harmonize import Harmonizer, default_methods
from .errors import (
    AmbiguousHarmonisationMethod,
    MissingHarmonisationYear,
    MissingHistoricalError,
)
from .methods import harmonize_factors


def harmonise_all(scenarios, history, harmonisation_year, overrides=None):
    """
    Harmonise all timeseries in ``scenarios`` to match ``history``

    Parameters
    ----------
    scenarios : :obj:`pd.DataFrame`
        :obj:`pd.DataFrame` containing the timeseries to be harmonised

    history : :obj:`pd.DataFrame`
        :obj:`pd.DataFrame` containing the historical timeseries to which
        ``scenarios`` should be harmonised

    harmonisation_year : int
        The year in which ``scenarios`` should be harmonised to ``history``

    overrides : :obj:`pd.DataFrame`
        If not provided, the default aneris decision tree is used. Otherwise,
        ``overrides`` must be a :obj:`pd.DataFrame` containing any
        specifications for overriding the default aneris methods. Each row
        specifies one override. The override method is specified in the
        "method" columns. The other columns specify which of the timeseries in
        ``scenarios`` should use this override by specifying metadata to match (
        e.g. variable, region). If a cell has a null value (evaluated using
        `pd.isnull()`) then that scenario characteristic will not be used for
        filtering for that override e.g. if you have a row with "method" equal
        to "constant_ratio", region equal to "World" and variable is null then
        all timeseries in the World region will use the "constant_ratio"
        method. In contrast, if you have a row with "method" equal to
        "constant_ratio", region equal to "World" and variable is
        "Emissions|CO2" then only timeseries with variable equal to
        "Emissions|CO2" and region equal to "World" will use the
        "constant_ratio" method.

    Returns
    -------
    :obj:`pd.DataFrame`
        The harmonised timeseries

    Notes
    -----
    This interface is nowhere near as sophisticated as aneris' other
    interfaces. It simply harmonises timeseries, it does not check sectoral
    sums or other possible errors which can arise when harmonising. If you need
    such features, do not use this interface.

    Raises
    ------
    MissingHistoricalError
        No historical data is provided for a given timeseries

    MissingHarmonisationYear
        A value for the harmonisation year is missing or is null in ``history``

    AmbiguousHarmonisationMethod
        ``overrides`` do not uniquely specify the harmonisation method for a
        given timeseries
    """
    # use groupby to maintain indexes, not sure if there's a better way because
    # this will likely be super slow
    res = scenarios.groupby(scenarios.index.names).apply(
        _harmonise_single, history, harmonisation_year, overrides
    )

    return res


def _harmonise_single(timeseries, history, harmonisation_year, overrides):
    assert timeseries.shape[0] == 1
    # unclear why we don't use pyam or scmdata for filtering
    mdata = {
        k: v for k, v in zip(timeseries.index.names, timeseries.index.to_list()[0])
    }

    variable = mdata["variable"]
    region = mdata["region"]

    hist_variable = history.index.get_level_values("variable") == variable
    hist_region = history.index.get_level_values("region") == region
    relevant_hist = history[hist_variable & hist_region]

    if relevant_hist.empty:
        error_msg = "No historical data for `{}` `{}`".format(region, variable)
        raise MissingHistoricalError(error_msg)

    if harmonisation_year not in relevant_hist:
        error_msg = "No historical data for year {} for `{}` `{}`".format(
            harmonisation_year, region, variable
        )
        raise MissingHarmonisationYear(error_msg)

    if relevant_hist[harmonisation_year].isnull().all():
        error_msg = "Historical data is null for year {} for `{}` `{}`".format(
            harmonisation_year, region, variable
        )
        raise MissingHarmonisationYear(error_msg)

    # convert units
    hist_unit = relevant_hist.index.get_level_values("unit").unique()[0]
    relevant_hist = _convert_units(
        relevant_hist, current_unit=hist_unit, target_unit=mdata["unit"]
    )
    # set index for rest of processing (as units are now consistent)
    relevant_hist.index = timeseries.index.copy()

    if overrides is not None:
        method = overrides.copy()
        for key, value in mdata.items():
            if key in method:
                method = method[(method[key] == value) | method[key].isnull()]

    if overrides is not None and method.shape[0] > 1:
        error_msg = (
            "Ambiguous harmonisation overrides for metdata `{}`, the "
            "following methods match: {}".format(mdata, method)
        )
        raise AmbiguousHarmonisationMethod(
            "More than one override for metadata: {}".format(mdata)
        )

    if overrides is None or method.empty:
        default, _ = default_methods(
            relevant_hist, timeseries, base_year=harmonisation_year
        )
        method_to_use = default.values[0]

    else:
        method_to_use = method["method"].values[0]

    return _harmonise_aligned(
        timeseries, relevant_hist, harmonisation_year, method_to_use
    )


def _convert_units(inp, current_unit, target_unit):
    # would be simpler using scmdata or pyam
    out = inp.copy()
    out.iloc[:, :] = (
        (out.values * unit_registry(current_unit)).to(target_unit).magnitude
    )
    out = out.reset_index("unit")
    out["unit"] = target_unit
    out = out.set_index("unit", append=True)

    return out


def _harmonise_aligned(timeseries, history, harmonisation_year, method):
    # seems odd that the methods are stored in a class instance
    harmonise_func = Harmonizer._methods[method]
    delta = _get_delta(timeseries, history, method, harmonisation_year)

    return harmonise_func(timeseries, delta, harmonize_year=harmonisation_year)


def _get_delta(timeseries, history, method, harmonisation_year):
    if method == "budget":
        return history

    offset, ratio = harmonize_factors(timeseries, history, harmonisation_year)
    if "ratio" in method:
        return ratio

    return offset
