from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import dask
import pandas as pd
import ptolemy as pt
import xarray as xr
from attrs import define, field
from pandas_indexing import isin

from .utils import Pathy, logger


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typing_extensions import Self


DEFAULT_INDEX = ("sector", "gas")


@dask.delayed
def verify_global_values(
    aggregated, tabular, output_variable, index, abstol=1e-8, reltol=1e-6
) -> pd.DataFrame | None:
    tab_df = tabular.groupby(level=index).sum().unstack("year")
    grid_df = aggregated.to_series().groupby(level=index).sum().unstack("year")
    grid_df, tab_df = grid_df.align(tab_df, join="inner")

    absdiff = abs(grid_df - tab_df)
    if (absdiff >= abstol + reltol * abs(tab_df)).any(axis=None):
        reldiff = (absdiff / tab_df).where(abs(tab_df) > 0, 0)
        logger().warning(
            f"Yearly global totals relative values between grids and global data for ({output_variable}) not within {reltol}:\n"
            f"{reldiff}"
        )
        return reldiff
    else:
        logger().info(
            f"Yearly global totals relative values between grids and global data for ({output_variable}) within tolerance"
        )
        return


@define
class GlobalIndexraster:
    """Pseudo indexraster mimicking ptolemy's indexraster for simple global aggregation"""

    dim: str = "country"

    @property
    def index(self):
        return ["World"]

    def grid(self, data):
        # Relies on xarray's broadcasting to spread out the data on the proxy file
        return data

    def aggregate(self, da):
        return da.sum(["lat", "lon"])


@define
class GriddingContext:
    indexraster_country: pt.IndexRaster
    indexraster_region: pt.IndexRaster
    cell_area: xr.DataArray
    index: Sequence[str] = field(factory=lambda: list(DEFAULT_INDEX))
    extra_spatial_dims: Sequence[str] = field(factory=lambda: ["level"])
    mean_time_dims: Sequence[str] = field(factory=lambda: ["month"])
    country_level: str = "country"
    year_level: str = "year"

    @property
    def indexrasters(self):
        return {
            "country": self.indexraster_country,
            "region": self.indexraster_region,
            "global": GlobalIndexraster(self.country_level),
        }

    @property
    def concat_dim(self):
        return self.index[0]

    @property
    def index_year(self):
        return [*self.index, self.year_level]

    @property
    def index_all(self):
        return [*self.index, self.country_level, self.year_level]


@define
class Gridded:
    data: xr.DataArray
    downscaled: pd.DataFrame
    proxy: Proxy
    meta: dict[str, str] = field(factory=dict)

    def verify(self, compute: bool = True):
        return self.proxy.verify_gridded(self.data, self.downscaled, compute=compute)

    def prepare_dataset(self, callback: Callable | None = None):
        name = self.proxy.name
        ds = self.data.to_dataset(name=name)

        if callback is not None:
            ds = callback(ds, name=name, **self.meta)

        return ds

    def fname(
        self,
        template_fn: str,
        directory: Pathy | None = None,
    ):
        meta = self.meta | dict(name=self.proxy.name)
        fn = template_fn.format(
            **{k: v.replace("_", "-").replace(" ", "-") for k, v in meta.items()}
        )
        if directory is not None:
            fn = Path(directory) / fn
        return fn

    def to_netcdf(
        self,
        template_fn: str,
        callback: Callable | None = None,
        encoding_kwargs: dict | None = None,
        directory: Pathy | None = None,
        compute: bool = True,
    ):
        ds = self.prepare_dataset(callback)
        encoding_kwargs = (
            ds[self.proxy.name].encoding
            | {
                "zlib": True,
                "complevel": 2,
            }
            | (encoding_kwargs or {})
        )
        return ds.to_netcdf(
            self.fname(template_fn, directory),
            encoding={self.proxy.name: encoding_kwargs},
            compute=compute,
        )


@define(slots=False)  # cached_property's need __dict__
class Proxy:
    # data is assumed to be given as a flux (beware: CEDS is in absolute terms)
    data: xr.DataArray
    levels: frozenset[str]
    context: GriddingContext
    name: str = "unnamed"

    @classmethod
    def from_files(
        cls,
        name: str,
        paths: Sequence[Pathy],
        levels: frozenset[str],
        context: GriddingContext,
        index_mappings: dict[str, dict[str, str]] | None = None,
    ) -> Self:
        if levels > (set(context.indexrasters) | {"global"}):
            raise ValueError(
                f"Variables need indexrasters for all levels: {', '.join(levels)}"
            )

        proxy = xr.concat(
            [
                xr.open_dataarray(path, chunks="auto", engine="h5netcdf").chunk(
                    {"lat": -1, "lon": -1}
                )
                for path in paths
            ],
            dim=context.concat_dim,
        )

        for dim in context.index:
            mapping = index_mappings.get(dim)
            if mapping is not None:
                proxy = (
                    proxy.rename({dim: f"proxy_{dim}"})
                    .sel({f"proxy_{dim}": xr.DataArray(mapping, dims=[dim])})
                    .drop_vars(f"proxy_{dim}")
                )

        return cls(proxy, levels, context, name)

    def reduce_dimensions(self, da):
        da = da.mean(self.context.mean_time_dims)
        spatial_dims = set(da.dims) & set(self.context.extra_spatial_dims)
        if spatial_dims:
            da = da.sum(spatial_dims)
        return da * self.context.cell_area

    @cached_property
    def weight(self):
        proxy_reduced = self.reduce_dimensions(self.data)

        return {
            level: self.context.indexrasters[level].aggregate(proxy_reduced).chunk(-1)
            for level in self.levels
        }

    def assert_single_pathway(self, downscaled):
        pathways = downscaled.pix.unique(
            downscaled.index.names.difference(self.context.index_all)
        )
        assert (
            len(pathways) == 1
        ), "`downscaled` is needed as a single scenario, but there are: {pathways}"
        return dict(zip(pathways.names, pathways[0]))

    def prepare_downscaled(self, downscaled):
        meta = self.assert_single_pathway(downscaled)
        downscaled = (
            downscaled.stack(self.context.year_level)
            .pix.semijoin(
                pd.MultiIndex.from_product(
                    [self.data.indexes[d] for d in self.context.index_year]
                ),
                how="inner",
            )
            .pix.project(self.context.index_all)
            .sort_index()
            .astype(self.data.dtype, copy=False)
        )
        downscaled.attrs.update(meta)
        return downscaled

    def verify_gridded(self, gridded, downscaled, compute: bool = True):
        scen = self.prepare_downscaled(downscaled)

        global_gridded = self.reduce_dimensions(gridded).sum(["lat", "lon"])
        diff = verify_global_values(
            global_gridded, scen, self.name, self.context.index_year
        )
        return diff.compute() if compute else diff

    def grid(self, downscaled: pd.DataFrame) -> Gridded:
        scen = self.prepare_downscaled(downscaled)

        def weighted(scen, weight):
            indexers = {
                dim: weight.indexes[dim].intersection(scen.pix.unique(dim))
                for dim in self.context.index
            }
            scen = xr.DataArray.from_series(scen).reindex(indexers, fill_value=0)
            weight = weight.reindex_like(scen)
            return (scen / weight).where(weight, 0).chunk()

        gridded = []
        for level in self.levels:
            indexraster = self.context.indexrasters[level]
            weight = self.weight[level]
            scen_ = scen.loc[isin(**{self.context.country_level: indexraster.index})]
            gridded_ = indexraster.grid(weighted(scen_, weight)).drop_vars(
                indexraster.dim
            )

            if gridded_.size > 0:
                gridded.append(self.data * gridded_)

        return Gridded(
            xr.concat(gridded, dim=self.context.concat_dim).assign_attrs(
                units=f"{scen.attrs['unit']} m-2"
            ),
            downscaled,
            self,
            scen.attrs,
        )
