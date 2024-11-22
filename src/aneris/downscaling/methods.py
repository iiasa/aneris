import logging
from typing import Union

import pandas_indexing.accessors  # noqa: F401
from pandas import DataFrame, Series
from pandas_indexing import semijoin

from ..utils import normalize
from .data import DownscalingContext
from .intensity_convergence import intensity_convergence  # noqa: F401


logger = logging.getLogger(__name__)


def base_year_pattern(
    model: DataFrame, hist: Union[Series, DataFrame], context: DownscalingContext
) -> DataFrame:
    """
    Downscales emission data using a base year pattern.

    Parameters
    ----------
    model : DataFrame
        model emissions for each world region and trajectory
    historic : DataFrame or Series
        historic emissions for each country and trajectory
    context : DownscalingContext
        settings for downscaling, like the regionmap

    Returns
    -------
    DataFrame:
        downscaled emissions for countries

    Notes
    -----
    1. All trajectories in `model` exist in `hist`
       a. `model` has the levels in `index` and "region"
       b. `hist` has the levels in `index` and "country"
    2. region mapping has two indices the first one is fine, the second coarse

    See also
    --------
    DownscalingContext
    """

    model = model.loc[:, context.year :]
    if isinstance(hist, DataFrame):
        hist = hist.loc[:, context.year]

    weights = (
        semijoin(hist, context.regionmap, how="right")
        .groupby(model.index.names, dropna=False)
        .transform(normalize)
    )

    # the semijoin in where should not be necessary, but pandas fails without it
    res = model.pix.multiply(weights, join="left")
    return res.where(semijoin(model != 0, res.index, how="right"), 0)


def simple_proxy(
    model: DataFrame,
    _hist: Union[Series, DataFrame],
    context: DownscalingContext,
    proxy_name: str,
) -> DataFrame:
    """
    Downscales emission data using the shares in a proxy scenario.

    Parameters
    ----------
    model : DataFrame
        model emissions for each world region and trajectory
    _hist : DataFrame or Series
        historic emissions for each country and trajectory.
        Unused for this method.
    context : DownscalingContext
        settings for downscaling, like the regionmap
    proxy_name : str
        name of the additional data to be used as proxy

    Returns
    -------
    DataFrame:
        downscaled emissions for countries

    See also
    --------
    DownscalingContext
    """

    model = model.loc[:, context.year :]

    proxy_data = context.additional_data[proxy_name]
    common_levels = [lvl for lvl in model.index.names if lvl in proxy_data.index.names]
    weights = (
        semijoin(proxy_data, context.regionmap)[model.columns]
        .groupby(common_levels + [context.region_level], dropna=False)
        .transform(normalize)
    )

    # the semijoin in where should not be necessary, but pandas fails without it
    res = model.pix.multiply(weights, join="left")
    return res.where(semijoin(model != 0, res.index, how="right"), 0)


def growth_rate(
    model: DataFrame,
    hist: Union[Series, DataFrame],
    context: DownscalingContext,
) -> DataFrame:
    """
    Downscales emission data using growth rates.

    Assumes growth rates in all sub regions are the same as in the macro_region

    Parameters
    ----------
    model : DataFrame
        model emissions for each world region and trajectory
    historic : DataFrame or Series
        historic emissions for each country and trajectory
    context : DownscalingContext
        settings for downscaling, like the regionmap

    Returns
    -------
    DataFrame:
        downscaled emissions for countries

    Notes
    -----
    1. All trajectories in `model` exist in `hist`
       a. `model` has the levels in `index` and "region"
       b. `hist` has the levels in `index` and "country"
    2. region mapping has two indices the first one is fine, the second coarse
    """

    model = model.loc[:, context.year :]
    if isinstance(hist, DataFrame):
        hist = hist.loc[:, context.year]

    cumulative_growth_rates = (model / model.shift(axis=1, fill_value=1)).cumprod(
        axis=1
    )

    weights = (
        cumulative_growth_rates.pix.multiply(
            semijoin(hist, context.regionmap, how="right"),
            join="left",
        )
        .groupby(model.index.names, dropna=False)
        .transform(normalize)
    )

    # the semijoin in where should not be necessary, but pandas fails without it
    res = model.pix.multiply(weights, join="left")
    return res.where(semijoin(model != 0, res.index, how="right"), 0)


def default_method_choice(
    traj,
    fallback_method="proxy_gdp",
    intensity_method="ipat_2100_gdp",
    luc_method="base_year_pattern",
    luc_sectors=None,
):
    """
    Default downscaling decision tree.
    """
    luc_sectors = luc_sectors or ("Agriculture", "LULUCF")

    # special cases
    if traj.h == 0:
        return fallback_method
    if traj.zero_m:
        return fallback_method

    if traj.get("sector", None) in luc_sectors:
        return luc_method

    return intensity_method
