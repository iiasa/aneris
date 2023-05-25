from functools import partial
from typing import Optional, Sequence

from pandas import DataFrame, Series
from pandas_indexing import concat, semijoin

from ..errors import MissingHistoricalError, MissingProxyError
from ..methods import default_methods
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
        "proxy_pop": partial(simple_proxy, proxy_name="gdp"),
    }

    def add_method(self, name, method):
        self._methods = self._methods | {name: method}

    def __init__(
        self,
        model: DataFrame,
        hist: DataFrame,
        year: int,
        region_mapping: Series,
        index: Sequence[str] = DEFAULT_INDEX,
        return_type=DataFrame,
        **additional_data: DataFrame,
    ):
        self.model = model
        self.hist = hist
        self.year = year
        self.return_type = return_type
        self.context = DownscalingContext(
            index,
            region_mapping,
            additional_data,
            country_level=region_mapping.index.name,
            region_level=region_mapping.name,
        )

        assert (
            hist[year].groupby(list(index) + [self.country_level]).count() <= 1
        ).all(), "Ambiguous history"

        missing_hist = (
            model.index.join(self.context.regionmap_index, how="left")
            .idx.project(list(index) + [self.country_level])
            .difference(hist.index.idx.project(list(index) + [self.country_level]))
        )
        if not missing_hist.empty:
            raise MissingHistoricalError(
                "History missing for variables/countries:\n"
                + missing_hist.to_frame().to_string(index=False)
            )

        # TODO Make configurable by re-using config just as in harmonizer
        self.intensity_method = "ipat_2100_gdp"
        self.luc_method = "base_year_pattern"

    @property
    def index(self):
        return self.context.index

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
                trajectory_index.idx.project(common_levels)
                .difference(proxy.index.idx.project(common_levels))
                .unique()
            )
            if not missing_proxy.empty:
                raise MissingProxyError(
                    f"The proxy data `{proxy_name}` is missing for the following "
                    "trajectories:\n" + missing_proxy.to_frame().to_string(index=False)
                )

            if not isinstance(proxy, DataFrame):
                return

            missing_years = self.model.columns.difference(proxy.columns)
            if not missing_years.empty:
                raise MissingProxyError(
                    f"The proxy data `{proxy_name}` is missing model year(s): "
                    + ", ".join(missing_years.astype(str))
                )

    def downscale(self, methods: Optional[Series] = None) -> DataFrame:
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

        return self.return_type(concat(downscaled))

    def methods(self, method_choice=None, overwrites=None):
        if method_choice is not None:
            method_choice = self.method_choice

        if method_choice is None:
            method_choice = default_method_choice

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
            method_choice=method_choice,
            intensity_method=self.intensity_method,
            luc_method=self.luc_method,
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
