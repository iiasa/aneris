import logging
from typing import Any, Optional, Union

import numpy as np
from pandas import DataFrame, Series, concat
from pandas_indexing import isin, semijoin
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from ..utils import normalize
from .data import DownscalingContext


logger = logging.getLogger(__name__)


class ConvergenceError(RuntimeError):
    pass


def make_affine_transform(x1, x2, y1=0.0, y2=1.0):
    """Returns an affine transform that maps `x1` to `y1` and `x2` to `y2`"""

    def f(x):
        return (y2 - y1) * (x - x1) / (x2 - x1) + y1

    return f


def compute_intensity(
    model: DataFrame, reference: DataFrame, convergence_year: int
) -> DataFrame:
    intensity = model.idx.divide(reference, join="left")

    model_years = model.columns
    if convergence_year > model_years[-1]:
        x2 = model_years[-1]
        x1 = x2 - 10 if x2 - 10 in model_years else model_years[-2]

        y1 = model[x1]
        y2 = model[x2]
        model_conv = (y2 * (y2 / y1) ** ((convergence_year - x2) / (x2 - x1))).where(
            y2 > 0, y2 + (y2 - y1) * (convergence_year - x2) / (x2 - x1)
        )

        y1 = reference[x1]
        y2 = reference[x2]
        reference_conv = y2 * (y2 / y1) ** ((convergence_year - x2) / (x2 - x1))

        intensity[convergence_year] = model_conv / reference_conv
    else:
        intensity = intensity.loc[:, :convergence_year]

    return intensity


def determine_scaling_parameter(
    alpha: Series,
    intensity_hist: Series,
    intensity: Series,
    reference: DataFrame,
    intensity_projection_linear: DataFrame,
    index: dict[str, Any],
    context: DownscalingContext,
) -> float:
    """Determine scaling parameter for negative exponential intensity model

    Gamma parameter for a single macro trajectory

    Parameters
    ----------
    alpha : Series
        Map from years to 0-1 range
    intensity_hist : Series
        Historic intensity of countries in base year
    intensity : Series
        Projected intensity of one worldregion/model
    reference : DataFrame
        Denominator of intensity
    intensity_projection_linear : DataFrame
        Per-country intensities previously determined by linear model
    index : dict[str, Any]
        Index levels of the full dataframe intensity
    context : DownscalingContext

    Returns
    -------
    gamma : float
    """
    levels = list(context.index) + [context.region_level]

    negative_at_start = intensity.iloc[0]
    if negative_at_start:
        raise ConvergenceError("Trajectory is fully negative")

    selector_levels = isin(**{k: v for k, v in index.items() if k in levels})
    reference = reference.loc[selector_levels]
    intensity_hist = intensity_hist.loc[selector_levels]

    intensity_projection_linear = intensity_projection_linear.loc[isin(**index)]

    # determine alpha_star, where projected emissions become negative
    res = root_scalar(
        interp1d(alpha, intensity),
        method="brentq",
        bracket=[0, 1],
    )
    if not res.converged:
        raise ConvergenceError(
            "Could not find alpha_star for which emissions cross into zero"
        )
    alpha_star = res.root
    year_star = make_affine_transform(0, 1, *intensity.index[[0, -1]])(alpha_star)

    # reference at alpha_star
    def at_alpha_star(df, alpha=alpha):
        return df.apply(
            (
                lambda s: interp1d(alpha, s, kind="slinear", fill_value="extrapolate")(
                    alpha_star
                )
            ),
            axis=1,
        )

    ref = at_alpha_star(reference, alpha=alpha[: len(reference.columns)])

    if intensity_projection_linear is not None:
        offset = (ref * at_alpha_star(intensity_projection_linear)).sum()
    else:
        offset = 0

    # determine gamma scaling parameter with which the sum of the weights from
    # the transformed model vanish at alpha_star
    def sum_weight_at_alpha_star(gamma):
        x0, x1 = intensity.iloc[[0, -1]]
        f = make_affine_transform(x0, x1, gamma, 1.0)
        inv_f = make_affine_transform(gamma, 1.0, x0, x1)

        return (
            ref * inv_f((f(x1) / f(intensity_hist)) ** alpha_star * f(intensity_hist))
        ).sum() + offset

    # Widen gamma_max until finding a sign flip in sum_weight_at_alpha_star
    gamma_min = 1.5
    gamma_max = 10 * gamma_min
    sum_weight_min = sum_weight_at_alpha_star(gamma_min)
    while sum_weight_min * sum_weight_at_alpha_star(gamma_max) > 0:
        if gamma_max >= 1e7:
            raise ConvergenceError(
                f"Exponential model does not converge to "
                f"{intensity.iloc[-1]} at {intensity.index[-1]}, "
                f"while guaranteeing zero emissions in {year_star}"
            )
        gamma_max *= 10

    res = root_scalar(
        sum_weight_at_alpha_star,
        method="brentq",
        bracket=[gamma_max / 10, gamma_max],
    )
    if not res.converged:
        raise ConvergenceError(
            "Could not determine scaling parameter gamma such that the weights"
            "from the exponential model vanish exactly with intensity"
        )

    gamma = res.root

    logger.debug(
        "Determined year(alpha_star) = %.2f, and gamma = %.2f",
        year_star,
        gamma,
    )

    return gamma


def negative_exponential_intensity_model(
    alpha: Series,
    intensity_hist: Series,
    intensity: DataFrame,
    reference: DataFrame,
    intensity_projection_linear: DataFrame,
    context: DownscalingContext,
    allow_fallback_to_linear: bool = True,
) -> DataFrame:
    """Create a per-country time-series of intensities w/ negative intensities

    Parameters
    ----------
    alpha : Series
        Map from years to 0-1 range
    intensity_hist : Series
        Historic intensity of countries in base year
    intensity : DataFrame
        Projected intensity of worldregion
    reference : DataFrame
        Denominator of intensity
    intensity_projection_linear : DataFrame
        _description_
    context : DownscalingContext

    Returns
    -------
    DataFrame
        _description_

    Raises
    ------
    ConvergenceError
        if it can not determine all scaling parameters
    """

    gammas = np.empty(len(intensity))

    for i, (index, intensity_traj) in enumerate(intensity.iterrows()):

        index = dict(zip(intensity.index.names, index))
        try:
            gammas[i] = determine_scaling_parameter(
                alpha,
                intensity_hist,
                intensity_traj,
                reference,
                intensity_projection_linear,
                index,
                context,
            )
        except ConvergenceError:
            if not allow_fallback_to_linear:
                raise
            gammas[i] = np.nan

    gammas = Series(gammas, intensity.index)

    f = make_affine_transform(intensity.iloc[:, 0], intensity.iloc[:, -1], gammas, 1.0)
    inv_f = make_affine_transform(
        gammas, 1.0, intensity.iloc[:, 0], intensity.iloc[:, -1]
    )

    intensity, intensity_hist = intensity.align(intensity_hist, join="left", axis=0)

    intensity_projection = DataFrame(
        inv_f(
            (f(intensity.values[:, -1]) / f(intensity_hist.values))[:, np.newaxis]
            ** alpha.values
            * f(intensity_hist.values)[:, np.newaxis]
        ),
        index=intensity_hist.index,
        columns=intensity.columns,
    ).where(
        gammas.notna(),
        intensity_growth_rate_model(intensity, intensity_hist),
    )

    return intensity_projection


def exponential_intensity_model(
    alpha: Series, intensity_hist: Series, intensity: DataFrame
) -> DataFrame:
    positive_intensity = intensity.iloc[:, -1] > 0
    if positive_intensity.all():
        f = inv_f = lambda x: x
    else:
        f = lambda x: np.where(positive_intensity, x, x + 1)
        inv_f = lambda x: np.where(positive_intensity, x, x - 1)

    intensity_hist = semijoin(intensity_hist, intensity.index, how="right")

    intensity_projection = DataFrame(
        inv_f(
            (f(intensity.values[:, -1]) / f(intensity_hist.values))[:, np.newaxis]
            ** alpha.values
            * f(intensity_hist.values)[:, np.newaxis]
        ),
        index=intensity_hist.index,
        columns=intensity.columns,
    )

    return intensity_projection


def linear_intensity_model(
    alpha: Series, intensity_hist: Series, intensity: DataFrame
) -> DataFrame:
    intensity_projection = DataFrame(
        (1 - alpha).values * intensity_hist.values[:, np.newaxis]
        + alpha.values * intensity.values[:, -1],
        index=intensity.index,
        columns=intensity.columns,
    )

    return intensity_projection


def intensity_growth_rate_model(
    intensity: DataFrame, intensity_hist: Series
) -> DataFrame:
    years_downscaling = intensity.columns
    intensity_projection = DataFrame(
        (
            1
            + (intensity.iloc[:, -1] / intensity_hist - 1)
            / (years_downscaling[-1] - years_downscaling[0])
        ).values[:, np.newaxis]
        ** np.arange(0, len(years_downscaling))
        * intensity_hist.values[:, np.newaxis],
        index=intensity_hist.index,
        columns=years_downscaling.rename("year"),
    )
    return intensity_projection


def intensity_convergence(
    model: DataFrame,
    hist: Union[Series, DataFrame],
    context: DownscalingContext,
    proxy_name: str = "gdp",
    convergence_year: Optional[int] = 2100,
    allow_fallback_to_linear: bool = True,
) -> DataFrame:
    """Downscales emission data using emission intensity convergence

    Parameters
    ----------
    model : DataFrame
        model emissions for each world region and trajectory
    historic : DataFrame or Series
        historic emissions for each country and trajectory
    context : DownscalingContext
        settings for downscaling, like the regionmap, and
        additional_data.
    proxy_name : str, default "gdp"
        name of the additional data used as a reference for intensity
        (intensity = model/reference)
    convergence_year : int, default 2100
        year of emission intensity convergence

    Returns
    -------
    DataFrame
        downscaled emissions for countries

    TODO
    ----
    We are assembling a dictionary, with intermediate results as `diagnostics`. Would be
    nice to give the user intuitive access.

    References
    ----------
    Gidden, M. et al. Global emissions pathways under different socioeconomic
    scenarios for use in CMIP6: a dataset of harmonized emissions trajectories
    through the end of the century. Geoscientific Model Development Discussions 12,
    1443–1475 (2019).
    """

    if isinstance(hist, DataFrame):
        hist = hist.iloc[:, -1]

    reference = semijoin(context.additional_data[proxy_name], context.regionmap_index)
    reference_region = reference.groupby(context.region_level).sum()
    hist = semijoin(hist, context.regionmap_index)

    intensity = compute_intensity(model, reference_region, convergence_year)
    intensity_hist = hist / reference.iloc[:, 0]

    alpha = make_affine_transform(intensity.columns[0], convergence_year)(
        intensity.columns
    )
    intensity_countries, intensity_hist = intensity.align(
        intensity_hist, join="left", axis=0
    )

    # use a linear model for sub-regions with an intensity below the convergence intensity
    low_intensity = intensity_hist <= intensity_countries.iloc[:, -1]

    if low_intensity.any():
        intensity_projection_linear = linear_intensity_model(
            alpha,
            intensity_hist.loc[low_intensity],
            intensity_countries.loc[low_intensity],
        )
        logger.debug(
            "Linear model was chosen for some trajectories: %s",
            ", ".join(intensity_hist.index[low_intensity]),
        )
    else:
        intensity_projection_linear = None
    del intensity_countries

    negative_convergence = intensity.iloc[:, -1] < 0
    if negative_convergence.any():
        negative_convergence_i = negative_convergence.index[negative_convergence]
        # sum does not work here. We need the individual per-country dimension
        negative_intensity_projection = negative_exponential_intensity_model(
            alpha,
            intensity_hist,
            intensity.loc[negative_convergence],
            reference,
            None
            if intensity_projection_linear is None
            else semijoin(
                intensity_projection_linear, negative_convergence_i, how="right"
            ),
            context
        )

    else:
        negative_intensity_projection = None

    if not negative_convergence.all():
        exponential_intensity_projection = exponential_intensity_model(
            alpha,
            intensity_hist.loc[~negative_convergence & ~low_intensity],
            intensity.loc[~negative_convergence],
        )
    else:
        exponential_intensity_projection = None

    intensity_projection = concat(
        filter(
            None,
            [
                exponential_intensity_projection,
                negative_intensity_projection,
                intensity_projection_linear,
            ],
        ),
        sort=False,
    ).reindex(index=intensity.index)

    intensity_projection = intensity_projection.loc[:, model.columns[-1]]

    if model.columns[-1] > intensity_projection.columns[-1]:
        # Extend modelled intensity projection beyond year_convergence
        intensity_projection = intensity_projection.reindex(
            columns=model.columns, method="ffill"
        )

    weights = (
        (reference * intensity_projection)
        .groupby(list(context.index) + [context.region_level])
        .transform(normalize)
    )
    return model.multiply(weights, how="left")
