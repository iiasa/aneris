from openscm_units import unit_registry
import pyam
import pandas as pd
from pandas_indexing import isin, semijoin


from .harmonize import Harmonizer, default_methods
from .errors import (
    AmbiguousHarmonisationMethod,
    MissingHarmonisationYear,
    MissingHistoricalError,
)
from .methods import harmonize_factors

def convert_units(fr, to, flabel='from', tlabel='to'):
    # this is a dumb way to do it and needs to be revised
    # but in short the idea is:
    # take fr and to dataframes and create a joined dataframe
    # on their variable and unit values
    # then for each value where units are different, do unit
    # conversion
    # you can't do blanket conversion, unfortunately, in case
    # there are variables which need to be converted differently
    def xform(x):
        return (
            x
            .timeseries()
            .reset_index()
            [['variable', 'unit']]
            .set_index('variable')
        )
    units = (
        xform(to)
        .join(xform(fr), how='left', lsuffix='_to', rsuffix='_fr')
        .drop_duplicates()
    )
    if units.isnull().values.any():
        missing = units[units.isnull().any(axis=1)]
        raise MissingHistoricalError(
            f'More {tlabel} than {flabel} values when trying to convert units:\n'
            f'{missing}'
            )
    # downselect to non-comparable units
    units = units[units.unit_to != units.unit_fr]
    # combine units that don't need changing with those that do
    fr_keep = fr.filter(variable=units.index, keep=False)
    fr_xform = fr.filter(variable=units.index)
    dfs = [] if fr_keep.empty else [fr_keep]
    for variable, row in units.iterrows():
        # pyam seems to not know about gas units... so we use scm_units
        factor = unit_registry(row.unit_fr).to(row.unit_to).magnitude
        dfs.append(
            fr_xform
            .filter(variable=variable)
            .convert_unit(row.unit_fr, to=row.unit_to, factor=factor)
        )
    return pyam.concat(dfs)

def _knead_overrides(overrides, scen, harm_idx):
    """Process overrides to get a form readable by aneris, supporting many different
    use cases

    Parameters
    ----------
    overrides : pd.DataFrame or pd.Series
    scen : pyam.IamDataFrame with data for single scenario and model instance
    """
    if overrides is None:
        return None

    # massage into a known format
    # check if no index and single value - this should be the override for everything
    if overrides.index.names == [None] and len(overrides['method']) == 1:
        _overrides = pd.Series(
            overrides['method'].values[0],
            index=pd.Index(scen.region, name=harm_idx[-1]), # only need to match 1 dim
            name='method',
            )
    # if data is provided per model and scenario, get those explicitly
    elif set(['model', 'scenario']).issubset(set(overrides.index.names)):
        _overrides = (
            overrides
            .loc[isin(model=scen.model, scenario=scen.scenario)]
            .droplevel(['model', 'scenario'])
        )
    # some of expected idx in cols, make it a multiindex
    elif isinstance(overrides, pd.DataFrame) and set(harm_idx) & set(overrides.columns): 
        idx = list(set(harm_idx) & set(overrides.columns))
        _overrides = overrides.set_index(idx)
    else:
        _overrides = overrides
    
    # do checks
    if _overrides.isnull().values.any():
        missing = _overrides[_overrides.isnull().any(axis=1)]
        raise AmbiguousHarmonisationMethod(
            f'Overrides are missing for provided data:\n'
            f'{missing}'
            )
    if _overrides.index.to_frame().isnull().values.any():
        missing = _overrides[_overrides.index.to_frame().isnull().any(axis=1)]
        raise AmbiguousHarmonisationMethod(
            f'Defined overrides are missing data:\n'
            f'{missing}'
            )
    if _overrides.index.duplicated().any():
        raise AmbiguousHarmonisationMethod(
            'Duplicated values for overrides:\n'
            f'{_overrides[_overrides.index.duplicated()]}'
        )

    return _overrides

def _check_data(hist, scen, harmonisation_year):
    check = ['region', 'variable']
    # @coroa - this may be a very slow way to do this check..
    def downselect(df):
        return (
            df
            .filter(year=harmonisation_year)
            ._data
            .reset_index()
            .set_index(check)
            .index
            .unique()
        )
    s = downselect(scen)
    h = downselect(hist)
    if h.empty:
        raise MissingHarmonisationYear(
            'No historical data in harmonization year'
        )

    if not s.difference(h).empty:
        raise MissingHistoricalError(
            'Historical data does not match scenario data in harmonization '
            f'year for\n {s.difference(h)}'
            )
    
    
# maybe this needs to live in pyam?
def harmonize_all2(scenarios, history, harmonisation_year, overrides=None):
    """
    Scenarios and History are pyam.IamDataFrames or pd.DataFrames which can be cast to them
    """
    year = harmonisation_year # TODO: change this to year
    sidx = scenarios.index # save in case we need to re-add extraneous indicies later
    as_pyam = isinstance(scenarios, pyam.IamDataFrame)
    if not as_pyam:
        scenarios = pyam.IamDataFrame(scenarios)
        history = pyam.IamDataFrame(history)

    dfs = []
    for (model, scenario) in scenarios.index:
        scen = scenarios.filter(model=model, scenario=scenario)
        hist = history.filter(
            region=scen.region, variable=scen.variable
            )
        _check_data(hist, scen, harmonisation_year)
        hist = convert_units(fr=hist, to=scen, flabel='history', tlabel='model')
        # need to convert to internal datastructure
        h = Harmonizer(
            scen.timeseries(), hist.timeseries(), 
            harm_idx=['variable', 'region']
            )
        # knead overrides
        _overrides = _knead_overrides(overrides, scen, harm_idx=['variable', 'region'])
        result = h.harmonize(year=year, overrides=_overrides)
        # need to convert out of internal datastructure
        dfs.append(
            result
            .assign(model=model, scenario=scenario)
            .set_index(['model', 'scenario'], append=True)
            .reorder_levels(pyam.utils.IAMC_IDX)
            )
    # realign indicies if more than standard IAMC_IDX were there originally
    result = pd.concat(dfs)
    result = (
        semijoin(result, sidx, how="right")
        .reorder_levels(sidx.names)
        )
    if as_pyam:
        result = pyam.IamDataFrame(result)
    return result


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
