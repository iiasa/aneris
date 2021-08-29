from openscm_units import unit_registry

from .harmonize import Harmonizer
from .methods import harmonize_factors


def harmonise_all(scenarios, history, harmonisation_year, overrides=None):
    # use groupby to maintain indexes, not sure if there's a better way because
    # this will likely be super slow
    res = scenarios.groupby(scenarios.index.names).apply(
        _harmonise_single,
        history,
        harmonisation_year,
        overrides
    )

    return res


def _harmonise_single(timeseries, history, harmonisation_year, overrides):
    assert timeseries.shape[0] == 1
    # unclear why we don't use pyam or scmdata for filtering
    mdata = {k: v for k, v in zip(timeseries.index.names, timeseries.index.to_list()[0])}


    relevant_hist = history[
        (history.index.get_level_values("variable") == mdata["variable"])
        & (history.index.get_level_values("region") == mdata["region"])
    ]


    # convert units
    hist_unit = relevant_hist.index.get_level_values("unit").unique()[0]
    # index for rest of processing (units updated by function below)
    relevant_hist.index = timeseries.index.copy()
    relevant_hist = _convert_units(relevant_hist, current_unit=hist_unit, target_unit=mdata["unit"])


    method = overrides.copy()
    for key, value in mdata.items():
        if key in method:
            method = method[method[key] == value]

    if method.shape[0] > 1:
        raise ValueError("More than one override for metadata: {}".format(mdata))

    method = method["method"].values[0]


    return _harmonise_aligned(timeseries, relevant_hist, harmonisation_year, method)


def _convert_units(inp, current_unit, target_unit):
    # would be simpler using scmdata or pyam
    out = inp.copy()
    out.iloc[:, :] = (out.values * unit_registry(current_unit)).to(target_unit).magnitude
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
