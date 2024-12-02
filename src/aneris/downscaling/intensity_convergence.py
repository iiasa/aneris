import logging
import time
from typing import Any, Optional, Union

import numpy as np
import pandas_indexing.accessors  # noqa: F401
from joblib import Parallel, delayed
from pandas import DataFrame, MultiIndex, Series, concat
from pandas_indexing import isin, semijoin
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from ..utils import normalize, skipempty
from .data import DownscalingContext


logger = logging.getLogger(__name__)


class ConvergenceError(RuntimeError):
    pass


def make_affine_transform(x1, x2, y1=0.0, y2=1.0):
    """
    Returns an affine transform that maps `x1` to `y1` and `x2` to `y2`
    """

    def f(x):
        beta = (x - x1) / (x2 - x1)
        return beta * y2 + (1 - beta) * y1

    return f


def make_affine_transform_pair(x1, x2, y1, y2):
    f = make_affine_transform(x1, x2, y1, y2)
    inv_f = make_affine_transform(y1, y2, x1, x2)
    return f, inv_f


def compute_intensity(
    model: DataFrame, reference: DataFrame, convergence_year: int
) -> DataFrame:
    intensity = model.pix.divide(reference, join="left")

    model_years = model.columns
    if convergence_year > model_years[-1]:
        # Assume intensity stays constant after model horizon
        intensity[convergence_year] = intensity.iloc[:, -1]

        ## extrapolation

        # x2 = model_years[-1]
        # x1 = x2 - 10 if x2 - 10 in model_years else model_years[-2]

        # y1 = model[x1]
        # y2 = model[x2]
        # model_conv = (y2 * (y2 / y1) ** ((convergence_year - x2) / (x2 - x1))).where(
        #     y2 > 0, y2 + (y2 - y1) * (convergence_year - x2) / (x2 - x1)
        # )

        # y1 = reference[x1]
        # y2 = reference[x2]
        # reference_conv = y2 * (y2 / y1) ** ((convergence_year - x2) / (x2 - x1))

        # intensity[convergence_year] = model_conv / reference_conv
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
    """
    Determine scaling parameter for negative exponential intensity model.

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
    negative_at_start = intensity.iloc[0] < 0
    if negative_at_start:
        raise ConvergenceError("Trajectory is fully negative")

    selector = isin(**index, ignore_missing_levels=True)
    reference = reference.loc[selector]
    intensity_hist = intensity_hist.loc[selector]
    intensity_projection_linear = intensity_projection_linear.loc[selector]

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

    if not intensity_projection_linear.empty:
        offset = (ref * at_alpha_star(intensity_projection_linear)).sum()
    else:
        offset = 0

    # determine gamma scaling parameter with which the sum of the weights from
    # the transformed model vanish at alpha_star
    def sum_weight_at_alpha_star(gamma):
        x0, x1 = intensity.iloc[[0, -1]]
        f, inv_f = make_affine_transform_pair(x0, x1, gamma, 1e-2)

        f_x1 = f(x1)
        f_hist = f(intensity_hist)
        return (ref * inv_f((f_x1 / f_hist) ** alpha_star * f_hist)).sum() + offset

    # Widen gamma_max until finding a sign flip in sum_weight_at_alpha_star
    gamma_min = 1e-2 * 1.01
    gamma_max = 10 * gamma_min
    sum_weight_min = sum_weight_at_alpha_star(gamma_min)

    err_msg = (
        f"Exponential model does not converge to "
        f"{intensity.iloc[-1]} at {intensity.index[-1]}, "
        f"while guaranteeing zero emissions in {year_star}"
    )
    if sum_weight_min < 0:
        bracket = [0, 1e-2 * 0.99]
        if sum_weight_at_alpha_star(0) * sum_weight_min >= 0:
            raise ConvergenceError(err_msg)
    else:
        while sum_weight_min * sum_weight_at_alpha_star(gamma_max) > 0:
            if gamma_max >= 1e20:
                raise ConvergenceError(err_msg)
            gamma_max *= 10

        bracket = [gamma_max / 10, gamma_max]

    res = root_scalar(sum_weight_at_alpha_star, method="brentq", bracket=bracket)
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


def _ts(s):
    if isinstance(s, (DataFrame, Series)):
        s = s.to_numpy()
    return np.asarray(s)[:, np.newaxis]


def negative_exponential_intensity_model(
    alpha: Series,
    intensity_hist: Series,
    intensity: DataFrame,
    reference: DataFrame,
    intensity_projection_linear: DataFrame,
    context: DownscalingContext,
    fallback_to_linear: bool = True,
    diagnostics: dict | None = None,
) -> DataFrame:
    """
    Create a per-country time-series of intensities w/ negative intensities.

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

    fallback_to_linear : bool, defaults to True
        Fallback to linear growth rate model
    diagnostics : dict, optional
        if a dict is passed in intermediates are added there

    Returns
    -------
    DataFrame
        _description_

    Raises
    ------
    ConvergenceError
        if it can not determine all scaling parameters
    """

    @delayed
    def determine_gamma(x):
        index, intensity_traj = x
        try:
            return determine_scaling_parameter(
                alpha,
                intensity_hist,
                intensity_traj,
                reference,
                intensity_projection_linear,
                dict(zip(intensity.index.names, index)),
                context,
            )
        except ConvergenceError:
            if not fallback_to_linear:
                raise
            return np.nan

    start = time.time()
    gammas = Series(
        Parallel(n_jobs=-1)(determine_gamma(x) for x in intensity.iterrows()),
        intensity.index,
    )
    logger.info("Determining scaling parameters took: %fs", time.time() - start)

    if gammas.isna().any():
        logger.warning(
            "Negative exponential intensity model has to fall back to"
            " simple growth rate model for %d out of %d trajectories",
            gammas.isna().sum(),
            len(intensity),
        )

    intensity_conv, intensity_hist_conv = intensity.loc[gammas.notna()].align(
        intensity_hist, join="left", axis=0, copy=False
    )
    gammas_conv = semijoin(gammas, intensity_conv.index, how="right")

    intensity_final = intensity_conv.iloc[:, -1]
    f, inv_f = make_affine_transform_pair(
        _ts(intensity_conv.iloc[:, 0]),
        _ts(intensity_final),
        _ts(gammas_conv),
        1e-2,
    )
    f_intensity_final = f(_ts(intensity_final))
    f_intensity_hist = f(_ts(intensity_hist_conv))
    intensity_projection = DataFrame(
        inv_f(
            (f_intensity_final / f_intensity_hist) ** alpha.to_numpy()
            * f_intensity_hist
        ),
        index=intensity_hist_conv.index,
        columns=intensity.columns,
    )

    if diagnostics is not None:
        diagnostics.update(
            gammas=gammas,
            f_intensity_final=f_intensity_final,
            f_intensity_hist=f_intensity_hist,
        )

    return concat(
        [
            intensity_projection,
            intensity_growth_rate_model(intensity.loc[gammas.isna()], intensity_hist),
        ],
        sort=False,
    )


def exponential_intensity_model(
    alpha: Series, intensity_hist: Series, intensity: DataFrame
) -> DataFrame:
    intensity_final = intensity.iloc[:, -1]
    positive_intensity = intensity_final > 0
    if positive_intensity.all():
        f = inv_f = lambda x: x
    else:
        offset = intensity.iloc[:, 0] / 1e4
        f = lambda x: x.pix.add(offset)
        inv_f = lambda x: x.pix.sub(offset)

    intensity_hist = semijoin(intensity_hist, intensity.index, how="right")

    intensity_projection = inv_f(
        DataFrame(
            (f(_ts(intensity_final)) / f(_ts(intensity_hist))) ** alpha.to_numpy()
            * f(_ts(intensity_hist)),
            index=intensity_hist.index,
            columns=intensity.columns,
        ),
    )

    return intensity_projection


def linear_intensity_model(
    alpha: Series, intensity_hist: Series, intensity: DataFrame
) -> DataFrame:
    intensity, intensity_hist = intensity.align(
        intensity_hist, join="left", copy=False, axis=0
    )
    intensity_projection = DataFrame(
        (1 - alpha).to_numpy() * _ts(intensity_hist)
        + alpha.to_numpy() * _ts(intensity.iloc[:, -1]),
        index=intensity.index,
        columns=intensity.columns,
    )

    return intensity_projection


@np.errstate(invalid="ignore")
def intensity_growth_rate_model(
    intensity: DataFrame, intensity_hist: Series
) -> DataFrame:
    intensity, intensity_hist = intensity.align(
        intensity_hist, join="left", axis=0, copy=False
    )

    years_downscaling = intensity.columns
    intensity_projection = DataFrame(
        _ts(
            1
            + (intensity.iloc[:, -1] / intensity_hist - 1)
            / (years_downscaling[-1] - years_downscaling[0])
        )
        ** np.arange(0, len(years_downscaling))
        * _ts(intensity_hist),
        index=intensity_hist.index,
        columns=years_downscaling.rename("year"),
    ).where(intensity_hist != 0, 0.0)
    return intensity_projection


def intensity_convergence(
    model: DataFrame,
    hist: Union[Series, DataFrame],
    context: DownscalingContext,
    proxy_name: str = "gdp",
    convergence_year: Optional[int] = 2100,
    fallback_to_linear: bool = True,
    diagnostics: dict | None = None,
) -> DataFrame:
    """
    Downscales emission data using emission intensity convergence.

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
    fallback_to_linear : bool, defaults to True
        Fallback to linear growth rate model
    diagnostics : dict, optional
        if a dict is passed in intermediates are added to it

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

    model = model.loc[:, context.year :]
    if isinstance(hist, DataFrame):
        hist = hist.loc[:, context.year]

    reference = semijoin(context.additional_data[proxy_name], context.regionmap)[
        model.columns
    ]
    reference_region = reference.groupby(
        reference.index.names.difference([context.country_level])
    ).sum()
    hist = semijoin(hist, context.regionmap)

    intensity = compute_intensity(model, reference_region, convergence_year)
    intensity_hist = hist / reference.iloc[:, 0]

    alpha = make_affine_transform(intensity.columns[0], convergence_year)(
        intensity.columns
    )
    intensity_countries, intensity_hist = intensity.align(
        intensity_hist, join="left", axis=0
    )
    intensity_idx = intensity_countries.index

    levels = list(model.index.names) + [context.country_level]
    empty_intensity = DataFrame(
        [],
        index=MultiIndex.from_arrays([[] for _ in levels], names=levels),
        columns=intensity.columns,
    )

    # use a linear model for countries with an intensity below the convergence intensity
    low_intensity = intensity_hist <= intensity_countries.iloc[:, -1]

    if low_intensity.any():
        intensity_projection_linear = linear_intensity_model(
            alpha,
            intensity_hist.loc[low_intensity],
            intensity_countries.loc[low_intensity],
        )
        logger.debug(
            "Linear model was chosen for some trajectories:\n%s",
            intensity_hist.index[low_intensity].to_frame().to_string(index=False),
        )
    else:
        intensity_projection_linear = empty_intensity
    del intensity_countries

    negative_convergence = intensity.iloc[:, -1] < 0
    if negative_convergence.any():
        negative_convergence_i = negative_convergence.index[negative_convergence]
        # sum does not work here. We need the individual per-country dimension

        negative_diagnostics = (
            diagnostics.setdefault("negative_exponential_intensity_model", dict())
            if diagnostics is not None
            else None
        )
        negative_intensity_projection = negative_exponential_intensity_model(
            alpha,
            intensity_hist.loc[~low_intensity],
            intensity.loc[negative_convergence],
            reference,
            semijoin(intensity_projection_linear, negative_convergence_i, how="inner"),
            context,
            fallback_to_linear=fallback_to_linear,
            diagnostics=negative_diagnostics,
        )

    else:
        negative_intensity_projection = empty_intensity

    if not negative_convergence.all():
        exponential_intensity_projection = exponential_intensity_model(
            alpha,
            intensity_hist.loc[~negative_convergence & ~low_intensity],
            intensity.loc[~negative_convergence],
        )
    else:
        exponential_intensity_projection = empty_intensity

    intensity_projection = concat(
        skipempty(
            exponential_intensity_projection,
            negative_intensity_projection,
            intensity_projection_linear,
        ),
        sort=False,
    ).reindex(index=intensity_idx)

    # if convergence year is past model horizon, intensity_projection is longer
    intensity_projection = intensity_projection.loc[:, : model.columns[-1]]

    if model.columns[-1] > intensity_projection.columns[-1]:
        # Extend modelled intensity projection beyond year_convergence
        intensity_projection = intensity_projection.reindex(
            columns=model.columns, method="ffill"
        )

    weights = (
        intensity_projection.pix.multiply(reference, join="left")
        .groupby(model.index.names, dropna=False)
        .transform(normalize)
    )

    if diagnostics is not None:
        diagnostics.update(
            reference=reference,
            intensity=intensity,
            intensity_projection=dict(
                exponential=exponential_intensity_projection,
                negative=negative_intensity_projection,
                linear=intensity_projection_linear,
            ),
            weights=weights,
        )

        res = model.pix.multiply(weights, join="left")
    return res.where(semijoin(model != 0, res.index, how="right"), 0)
