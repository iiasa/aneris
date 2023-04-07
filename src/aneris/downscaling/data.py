from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union

from pandas import DataFrame, MultiIndex, Series


@dataclass
class DownscalingContext:
    """
    Context in which downscaling needs to happen.

    Attributes
    ----------
    index: sequence of str
        index levels that differentiate trajectories
    regionmap: Series
        map from countries to regions
    additional_data: dict, default {}
        named `DataFrame`s or `Series` the methods need as proxies
    country_level: str, default "country"
        name of the fine index level
    region_level: str, default "region"
        name of the coarse index level

    Notes
    -----
    Passed as context argument to each downscaling method
    """

    index: Sequence[str]
    regionmap: Series
    additional_data: dict[str, Union[Series, DataFrame]] = field(default_factory=dict)
    country_level: str = "country"
    region_level: str = "region"

    @property
    def regionmap_index(self):
        return MultiIndex.from_arrays(
            [self.regionmap.index, self.regionmap.values],
            names=[self.country_level, self.region_level],
        )
