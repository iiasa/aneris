from typing import Optional, Sequence, Callable
from functools import partial

from pandas import DataFrame, Series, Index
from pandas_indexing import projectlevel, semijoin

from ..methods import default_methods
from .data import DownscalingContext
from .methods import base_year_pattern, growth_rate, intensity_convergence


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
    }

    def add_method(self, name, method):
        self._methods = self._methods | {name: method}

    def __init__(
        self,
        model: DataFrame,
        hist: DataFrame,
        region_mapping: Series,
        index: Sequence[str] = DEFAULT_INDEX,
        return_type=DataFrame,
        **additional_data: DataFrame,
    ):
        self.model = model
        self.hist = hist
        self.region_mapping = region_mapping
        self.index = index
        self.return_type = return_type
        self.additional_data = additional_data

        self.region_level = self.region_mapping.name
        self.country_level = self.region_mapping.index.name

        assert (
            hist.groupby(list(index) + [self.region_level]).count() <= 1
        ).all(), "More than one hist"
        assert (
            projectlevel(model.index, list(index) + [self.region_level])
            .difference(projectlevel(hist.index, list(index) + [self.region_level]))
            .empty
        ), "History missing for some"

        # TODO Make configurable by re-using config just as in harmonizer
        self.intensity_method = "ipat_2100_gdp"
        self.linear_method = "base_year_pattern"

    def downscale(self, methods: Optional[Series] = None) -> DataFrame:
        """Downscale aligned model data from historical data, and socio-economic scenario

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

        # Check that data contains what is needed for all methods in use, ie. inspect partial keywords

        return self.return_type(downscaled)

    def methods(self, method_choice=None, overwrites=None):
        if method_choice is not None:
            method_choice = self.method_choice

        if method_choice is None:
            method_choice = default_method_choice

        hist_agg = (
            semijoin(self.hist, self.context.regionmap_index)
            .groupby(list(self.index) + [self.country_level], dropna=False)
            .sum()
        )
        methods = default_methods(
            projectlevel(self.model, list(self.index) + [self.region_level]),
            hist_agg,
            method_choice=method_choice,
            intensity_method=self.intensity_method,
            linear_method=self.linear_method,
        )

        if isinstance(overwrites, str):
            return Series(overwrites, methods.index)
        elif isinstance(overwrites, dict):
            overwrites = Series(overwrites).rename_axis("sector")

        return (
            semijoin(overwrites, methods.index, how="right")
            .fillna(methods)
            .rename("method")
        )

    @property
    def context(self):
        return DownscalingContext(
            self.index,
            self.region_mapping,
            self.additional_data,
            self.country_level,
            self.region_level,
        )


def default_method_choice(traj, intensity_method, linear_method):
    """Default downscaling decision tree"""

    # special cases
    if traj.h == 0:
        return linear_method
    if traj.zero_m:
        return linear_method

    if traj.get("sector", None) in ("Agriculture", "LULUCF"):
        return linear_method

    return intensity_method
