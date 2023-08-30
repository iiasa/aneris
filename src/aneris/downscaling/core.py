from functools import partial
from typing import Optional, Sequence

import pandas_indexing.accessors  # noqa: F401
from pandas import DataFrame, Series
from pandas_indexing import concat, semijoin

from ..errors import MissingHistoricalError, MissingProxyError
from ..methods import default_methods
from ..utils import logger
from .data import DownscalingContext
from .methods import (
    base_year_pattern,
    default_method_choice,
    growth_rate,
    intensity_convergence,
    simple_proxy,
)


DEFAULT_INDEX = ("sector", "gas")


class Downscaler:
    _methods = {
        "ipat_2100_gdp": partial(
            intensity_convergence, convergence_year=2100, proxy_name="gdp"
        ),
        "ipat_2150_pop": partial(
            intensity_convergence, convergence_year=2150, proxy_name="pop"
        ),
        "base_year_pattern": base_year_pattern,
        "growth_rate": growth_rate,
        "proxy_gdp": partial(simple_proxy, proxy_name="gdp"),
        "proxy_pop": partial(simple_proxy, proxy_name="pop"),
    }

    def add_method(self, name, method):
        self._methods = self._methods | {name: method}

    def __init__(
        self,
        model: DataFrame,
        hist: DataFrame,
        year: int,
        region_mapping: Series,
        luc_sectors: Sequence[str] = [],
        index: Sequence[str] = DEFAULT_INDEX,
        method_choice: Optional[callable] = None,
        return_type=DataFrame,
        **additional_data: DataFrame,
    ):
        self.model = model
        self.hist = hist
        self.return_type = return_type
        self.context = DownscalingContext(
            index,
            year,
            region_mapping,
            additional_data,
            country_level=region_mapping.index.name,
            region_level=region_mapping.name,
        )

        assert (
            hist[self.year].groupby(list(index) + [self.country_level]).count() <= 1
        ).all(), "Ambiguous history"

        missing_hist = (
            model.index.join(self.context.regionmap_index, how="left")
            .pix.project(list(index) + [self.country_level])
            .difference(hist.index.pix.project(list(index) + [self.country_level]))
        )
        if not missing_hist.empty:
            raise MissingHistoricalError(
                "History missing for variables/countries:\n"
                + missing_hist.to_frame().to_string(index=False, max_rows=100)
            )

        # TODO Make configurable by re-using config just as in harmonizer
        self.fallback_method = None
        self.intensity_method = None
        self.luc_method = None
        self.method_choice = method_choice
        self.luc_sectors = luc_sectors

    @property
    def index(self):
        return self.context.index

    @property
    def year(self):
        return self.context.year

    @property
    def region_mapping(self):
        return self.context.regionmap

    @property
    def additional_data(self):
        return self.context.additional_data

    @property
    def country_level(self):
        return self.context.country_level

    @property
    def region_level(self):
        return self.context.region_level

    def check_proxies(self, methods: Series) -> None:
        """
        Checks proxies required for chosen `methods`

        Parameters
        ----------
        methods : Series
            Methods to be used for each trajectory

        Raises
        ------
        MissingProxyError
            if a required proxy is missing or incomplete
        """
        for method in methods.unique():
            proxy_name = getattr(self._methods[method], "keywords", {}).get(
                "proxy_name"
            )
            if proxy_name is None:
                continue

            proxy = self.additional_data.get(proxy_name)
            if proxy is None:
                raise MissingProxyError(
                    f"Downscaling method `{method}` requires the additional data"
                    f" `{proxy_name}`"
                )

            trajectory_index = methods.index[methods == method]

            # trajectory index typically has the levels model, scenario, region, sector,
            # gas, while proxy data is expected on country level (and probably no model,
            # scenario dependency, but potentially)
            proxy = semijoin(proxy, self.context.regionmap_index, how="right")

            common_levels = [
                lvl for lvl in trajectory_index.names if lvl in proxy.index.names
            ]
            missing_proxy = (
                trajectory_index.pix.project(common_levels)
                .difference(proxy.index.pix.project(common_levels))
                .unique()
            )
            if not missing_proxy.empty:
                raise MissingProxyError(
                    f"The proxy data `{proxy_name}` is missing for the following "
                    "trajectories:\n"
                    + missing_proxy.to_frame().to_string(index=False, max_rows=100)
                )

            if not isinstance(proxy, DataFrame):
                return

            missing_years = self.model.columns.difference(proxy.columns)
            if not missing_years.empty:
                raise MissingProxyError(
                    f"The proxy data `{proxy_name}` is missing model year(s): "
                    + ", ".join(missing_years.astype(str))
                )

    def downscale(
        self, methods: Optional[Series] = None, check_result: bool = True
    ) -> DataFrame:
        """
        Downscale aligned model data from historical data, and socio-economic
        scenario.

        Notes
        -----
        model.index contains at least the downscaling index levels, but also any other
        levels.

        hist.index contains at least the downscaling index levels other index levels are
        allowed, but only one value per downscaling index value.

        Parameters
        ----------
        methods : Series Methods to apply

        check_result : bool, default True
            Check whether the downscaled trajectories sum up to the regional totals
        """

        if methods is None:
            methods = self.methods()

        hist_ext = semijoin(self.hist, self.context.regionmap_index, how="right")
        self.check_proxies(methods)

        downscaled = []
        method_groups = methods.index.groupby(methods)
        for method, trajectory_index in method_groups.items():
            hist = semijoin(hist_ext, trajectory_index, how="right")
            model = semijoin(self.model, trajectory_index, how="right")

            downscaled.append(self._methods[method](model, hist, self.context))

        downscaled = concat(downscaled)
        if check_result:
            self.check_downscaled(downscaled)

        return self.return_type(downscaled)

    def check_downscaled(self, downscaled, rtol=1e-05, atol=1e-08):
        downscaled = (
            downscaled.groupby(self.model.index.names, dropna=False)
            .sum()
            .rename_axis(columns="year")
            .stack()
        )
        model = self.model.rename_axis(columns="year").stack()
        diff = downscaled - model
        diff_exceeded = abs(diff) + rtol * abs(model) > atol
        if diff_exceeded.any():
            logger().warning(
                "Difference thresholds exceeded for a few trajectories:\n%s",
                DataFrame(dict(model=model, downscaled=downscaled, diff=diff))
                .loc[diff_exceeded]
                .to_string(),
            )

    def methods(self, method_choice=None, overwrites=None):
        if method_choice is None:
            method_choice = self.method_choice

        if method_choice is None:
            method_choice = default_method_choice

        kwargs = {
            "method_choice": method_choice,
            "fallback_method": self.fallback_method,
            "intensity_method": self.intensity_method,
            "luc_method": self.luc_method,
            "luc_sectors": self.luc_sectors,
        }

        hist_agg = (
            semijoin(self.hist, self.context.regionmap_index, how="right")
            .groupby(list(self.index) + [self.region_level], dropna=False)
            .sum()
        )
        methods, meta = default_methods(
            semijoin(hist_agg, self.model.index, how="right").reorder_levels(
                self.model.index.names
            ),
            self.model,
            self.year,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        if overwrites is None:
            return methods
        elif isinstance(overwrites, str):
            return Series(overwrites, methods.index)
        elif isinstance(overwrites, dict):
            overwrites = Series(overwrites).rename_axis("sector")

        return (
            semijoin(overwrites, methods.index, how="right")
            .combine_first(methods)
            .rename("method")
        )
