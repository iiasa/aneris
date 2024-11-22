from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Union

from pandas import DataFrame, MultiIndex, Series


@dataclass
class DownscalingContext:
    """
    Context in which downscaling needs to happen.

    Attributes
    ----------
    index : sequence of str
        index levels that differentiate trajectories
    year : int
        base year for downscaling
    regionmap : MultiIndex
        map from fine to coarse level
        (there can be overlapping coarse levels)
    additional_data : dict, default {}
        named `DataFrame`s or `Series` the methods need as proxies

    Derived attributes
    -------------------
    country_level : str, default "country"
        name of the fine index level
    region_level : str, default "region"
        name of the coarse index level

    Notes
    -----
    Passed as context argument to each downscaling method
    """

    index: Sequence[str]
    year: int
    regionmap: MultiIndex
    additional_data: Mapping[str, Union[Series, DataFrame]] = field(
        default_factory=dict
    )

    @property
    def country_level(self) -> str:
        return self.regionmap.names[0]

    @property
    def region_level(self) -> str:
        return self.regionmap.names[1]

    @staticmethod
    def to_regionmap(region_mapping: Union[Series, MultiIndex]):
        if isinstance(region_mapping, MultiIndex):
            return region_mapping

        return MultiIndex.from_arrays(
            [region_mapping.index, region_mapping.values],
            names=[region_mapping.index.name, region_mapping.name],
        )
