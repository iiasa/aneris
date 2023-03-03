import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, Union

from pandas import DataFrame, MultiIndex, Series
from pandas_indexing import semijoin

from ..utils import normalize
from .data import DownscalingContext
from .intensity_convergence import intensity_convergence


logger = logging.getLogger(__name__)



def base_year_pattern(
    model: DataFrame, hist: Union[Series, DataFrame], context: DownscalingContext
) -> DataFrame:
    """Downscales emission data using a base year pattern

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

    if isinstance(hist, DataFrame):
        hist = hist.iloc[:, -1]

    weights = (
        semijoin(hist, context.regionmap_index)
        .groupby(list(context.index) + [context.region_level])
        .transform(normalize)
    )

    return model.idx.multiply(weights, join="left")


def growth_rate(
    model: DataFrame,
    hist: Union[Series, DataFrame],
    context: DownscalingContext,
) -> DataFrame:
    """Downscales emission data using growth rates

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

    if isinstance(hist, DataFrame):
        hist = hist.iloc[:, -1]

    cumulative_growth_rates = (model / model.shift(axis=1, fill_value=1)).cumprod(
        axis=1
    )

    weights = (
        cumulative_growth_rates.idx.multiply(
            semijoin(hist, context.regionmap_index),
            join="left",
        )
        .groupby(list(context.index) + [context.region_level])
        .transform(normalize)
    )

    return model * weights


