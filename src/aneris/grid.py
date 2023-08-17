import warnings
from contextlib import contextmanager
from functools import reduce
from itertools import repeat
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import dask
import pandas_indexing.accessors  # noqa: F401
import ptolemy as pt
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from pandas import DataFrame, MultiIndex, Series
from pandas_indexing import isin, semijoin
from xarray import DataArray

from .errors import MissingCoordinateValue, MissingDimension, MissingLevels
from .utils import country_name, logger


DEFAULT_INDEX = ("sector", "gas", "year")


class Gridder:
    def __init__(
        self,
        data: DataFrame,
        idxraster: DataArray,
        proxy_cfg: DataFrame,
        index: Sequence[str] = DEFAULT_INDEX,
        index_mappings: Optional[dict[str, dict[str, str]]] = None,
        country_level: str = "country",
        output_dir: Optional[Union[Path, str]] = None,
    ):
        """
        Prepare gridding data.

        Parameters
        ----------
        data : DataFrame or Series
            Tabular data (should contain `country_level` and `index` levels)
            If a DataFrame is given, it expects 'year's on the columns.
        idxraster : DataArray
            Rasterized country map (should sum to 1 over `country_level`)
        proxy_cfg : DataFrame
            Configuration of proxies with the columns:
            "name", "path", "template", "as_flux", "global_only", "separate_shares"
        index : Sequence[str], optional
            level names on which to align between tabular data and proxies, by default
            DEFAULT_INDEX
        index_mappings : Optional[dict[str, dict[str, str]]], optional
            Mapping from proxy index coordinate to data values, by default None
        country_level : str, optional
            level or dimension name for countries, by default "country"
        output_dir : str or Path, optional
            directory in which to create gridded proxies
            if omitted or None, files are created in the current working directory
        """
        if isinstance(data, DataFrame):
            data = data.rename_axis(columns="year").stack()
        self.data = data

        self.idxraster = idxraster

        if isinstance(proxy_cfg, (list, tuple)):
            proxy_cfg = DataFrame(dict(path=Series(proxy_cfg).map(Path)))
        if isinstance(proxy_cfg, DataFrame):
            proxy_cfg = proxy_cfg.copy(deep=False)
            if "name" not in proxy_cfg.columns:
                proxy_cfg["name"] = proxy_cfg["path"].map(lambda p: p.stem)
            if "template" not in proxy_cfg.columns:
                proxy_cfg["template"] = (
                    "{gas}-em-" + proxy_cfg["name"] + "-{model}-{scenario}"
                )
            if "as_flux" not in proxy_cfg.columns:
                proxy_cfg["as_flux"] = True
            if "global_only" not in proxy_cfg.columns:
                proxy_cfg["global_only"] = False
            if "separate_shares" not in proxy_cfg.columns:
                proxy_cfg["separate_shares"] = False
        self.proxy_cfg = proxy_cfg

        self.index = list(index)
        self.spatial_dims = ["lat", "lon", "level"]
        self.mean_time_dims = ["month"]
        self.index_mappings = index_mappings if index_mappings is not None else dict()
        self.country_level = country_level

        self.output_dir = Path.cwd() if output_dir is None else Path(output_dir)

    def check(
        self, strict_proxy_data: bool = False, global_label: str = "World"
    ) -> None:
        """
        Check levels and dimensions of gridding data.

        Parameters
        ----------
        strict_proxy_data : bool, default True
            If true, proxy data must align with tabular data. If false, proxy
            data can have additional data than is provided in tabular data
            (e.g., additional years)
        global_label : str, default "World"
            The regional label applied to global data which should not be
            checked against country proxy data

        Raises
        ------
        MissingLevels
            If `data` is missing levels
        MissingDimension
            If `idxraster` or a proxy is missing dimensions
        MissingCoordinateValue
            If tabular and spatial data is misaligned
        """
        # Check data
        missing_levels = {self.country_level, *self.index}.difference(
            self.data.index.names
        )
        if missing_levels:
            raise MissingLevels(
                "Tabular `data` must have `country_level` and `index` levels, "
                "but is missing: " + ", ".join(missing_levels)
            )

        # Check idxraster
        idxr_missing_dims = {self.country_level, "lat", "lon"}.difference(
            self.idxraster.dims
        )
        if idxr_missing_dims:
            raise MissingDimension(
                "idx_raster missing dimensions: " + ", ".join(idxr_missing_dims)
            )

        # Check data and idxraster alignment
        countries_data = set(self.data.idx.unique(self.country_level)) - set(
            [global_label]
        )
        countries_idx = set(self.idxraster.indexes[self.country_level])
        missing_from_idxraster = countries_data - countries_idx
        if missing_from_idxraster:
            raise MissingCoordinateValue(
                f"`idxraster` missing countries ('{self.country_level}'): "
                + ", ".join(country_name(x) for x in missing_from_idxraster)
            )
        missing_from_data = countries_idx - countries_data
        if missing_from_data:
            logger().warning(
                f"Tabular `data` missing countries ('{self.country_level}'): "
                + ", ".join(country_name(x) for x in missing_from_data)
            )

        # Check proxies and alignment with data
        data_index = self.data.idx.unique(self.index)

        def get_index(dim):
            idx = proxy.indexes[dim]
            mapping = self.index_mappings.get(dim)
            if mapping is not None:
                idx = idx.map(mapping)
            return idx

        proxy_index = []
        for proxy_cfg in self.proxy_cfg.itertuples():
            with xr.open_dataset(proxy_cfg.path) as proxy:
                proxy_missing_dims = {"lat", "lon", *self.index}.difference(proxy.dims)
                if proxy_missing_dims:
                    raise MissingDimension(
                        f"Proxy {proxy_cfg.name} missing dimensions: "
                        + ", ".join(proxy_missing_dims)
                    )

                index = MultiIndex.from_product([get_index(dim) for dim in self.index])
                missing_from_data = index.difference(data_index)
                if not missing_from_data.empty:
                    msg = (
                        f"Proxy '{proxy_cfg.name}' has values missing from `data`:\n"
                        + missing_from_data.to_frame().to_string(index=False)
                    )
                    if strict_proxy_data:
                        raise MissingCoordinateValue(msg)
                    else:
                        warnings.warn(msg)

                proxy_index.append(index)

        def concat(objs):
            return reduce(lambda x, y: x.append(y), objs)

        proxy_index = concat(proxy_index)
        missing_from_proxy = data_index.difference(proxy_index)
        if not missing_from_proxy.empty:
            raise MissingCoordinateValue(
                "None of the configured proxies provides:\n"
                + missing_from_proxy.to_frame().to_string(index=False)
            )

    @contextmanager
    def open_and_normalize_proxy(self, proxy_cfg, chunk_proxy_dims={}):
        with xr.open_dataarray(
            proxy_cfg.path,
            chunks=dict(
                **dict(zip(self.index, repeat(1))), 
                **chunk_proxy_dims
                ),
        ) as proxy:
            for idx in self.index:
                mapping = self.index_mappings.get(idx)
                if mapping is not None:
                    proxy[idx] = proxy.indexes[idx].map(mapping)

            # TODO: this maybe isn't needed anymore with 'World' included in idxraster
            #       but need to confirm 'World' is also in the proxy rasters
            separate = proxy if proxy_cfg.global_only else self.idxraster * proxy
            # separate = self.idxraster * proxy

            # NB: this only preserves seasonality if years and months are
            #     separate dimensions in the proxy raster. If instead they are
            #     combined into a single 'time' dimension, seasonality is lost.
            sum_spatial_dims = list(set(separate.dims).intersection(self.spatial_dims))
            normalized = separate / separate.mean(self.mean_time_dims).sum(
                sum_spatial_dims
            )

            if proxy_cfg.as_flux:
                lat_areas_in_m2 = xr.DataArray.from_series(
                    pt.cell_area_from_file(proxy)
                )
                normalized = normalized / lat_areas_in_m2

            yield normalized

    # TODO: iter_levels was added because some trajectories can have different
    # downscaling methods applied? E.g., for burning emissions, proxy_gdp and
    # ipat are both used, causing the gridding process to be called twice in
    # `for iter_vals in tabular.idx.unique(iter_levels)`
    #
    # TODO: chunk_proxy_dims can in principle be moved into Gridder.proxy_cfg,
    # but requires supporting lists, so need to decide how to deal with that
    def grid(
        self,
        skip_check: bool = False,
        chunk_proxy_dims: Mapping[str, int] = {},
        iter_levels: Sequence[str] = [],
        write: bool = True,  # TODO: make docs
        share_dims: Sequence[str] = ["sector"],  # TODO: make docs
        verify_output: bool = False,  # TODO: make docs
    ) -> None:
        """
        Grid data onto configured proxies.

        Parameters
        ----------
        skip_check : bool, default False
            If set, skips structural and alignment checks
        chunk_proxy_dims : Sequence[str], default []
            Additional dimensions to chunk when opening proxy files
        iter_levels : Sequence[str], default []
            Explicit levels over which to iterate (e.g., model and scenario)
        """
        if not skip_check:
            self.check()

        iter_levels = iter_levels or self.data.index.names.difference(
            self.index + [self.country_level]
        )
        ret = []
        for proxy_cfg in self.proxy_cfg.itertuples():
            logger().info("Collecting tasks for proxy %s", proxy_cfg.name)

            with self.open_and_normalize_proxy(proxy_cfg, chunk_proxy_dims) as proxy:
                write_tasks = []

                proxy_index = MultiIndex.from_product(
                    [proxy.indexes[dim] for dim in self.index]
                )
                # dropna is required when data is allowed to have less dimension
                # values than proxy (e.g., fewer years)
                tabular = semijoin(self.data, proxy_index, how="inner")

                for iter_vals in tabular.idx.unique(iter_levels):
                    iter_ids = dict(zip(iter_levels, iter_vals))
                    logger().info("Adding tasks for %s", iter_ids)
                    data = DataArray.from_series(
                        tabular.loc[isin(**iter_ids)].droplevel(iter_levels)
                    )
                    gridded = (data * proxy).sum(self.country_level)

                    if verify_output:
                        write_tasks.append(self.verify_output(tabular, gridded))

                    write_tasks.append(
                        self.compute_output(
                            proxy_cfg,
                            gridded,
                            data.indexes,
                            iter_ids,
                            write=write,
                            share_dims=share_dims,
                        )
                    )

                if write:
                    with ProgressBar():
                        dask.compute(write_tasks)
                else:
                    ret.append(write_tasks)

        return ret

    def verify_output(
        self,
        tabular,
        gridded,
    ):
        # TODO: figure out correct message here
        # ids = {dim: index[0] for dim, index in indexes.items() if len(index) == 1}
        # logger().info(f"Veryifying output for {ids}")

        # TODO: this is complex and can be given to us by the user?
        # the point of this function is to compute global totals across
        # self.index (nominally sector, gas, year), and compare with
        # the same values summed up in the original tabular data provided
        # to confirm that gridded values comport with provided global totals
        sum_spatial_dims = list(set(gridded.dims).intersection(self.spatial_dims))
        droplevel = list(
            set(gridded.dims).difference(
                set(self.index + self.spatial_dims + self.mean_time_dims)
            )
        )
        grid_df = (
            (xr.DataArray(pt.cell_area_from_file(gridded)) * gridded)
            .mean(dim=self.mean_time_dims)
            .sum(dim=sum_spatial_dims)
            .to_dataframe(name="emissions")
            .unstack("year")
            .droplevel(droplevel)["emissions"]
        )
        tab_df = (
            semijoin(tabular, grid_df.index, how="inner")
            .groupby(level=grid_df.index.names)
            .sum()[grid_df.columns]
        )
        rel_diff = (grid_df - tab_df).abs() / tab_df
        lim = 1e-4
        if not (rel_diff < lim).all().all():
            logger().warning(
                f"Yearly global totals not within {lim} relative values:\n"
                f"{rel_diff}"
            )
        return rel_diff

    def compute_output(
        self,
        proxy_cfg,
        gridded: DataArray,
        indexes,
        iter_ids,
        write=True,
        share_dims=["sector"],
        comp=dict(zlib=True, complevel=5),
    ):
        # TODO: need to add attr definitions and dimension bounds
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ids = {dim: index[0] for dim, index in indexes.items() if len(index) == 1}
        fname = (
            proxy_cfg.template.format(name=proxy_cfg.name, **ids, **iter_ids).replace(
                " ", "__"
            )
            + ".nc"
        )
        path = self.output_dir / fname
        logger().info(f"Writing to {path}")
        if not proxy_cfg.separate_shares:
            gridded = gridded.to_dataset(name=proxy_cfg.name)
            if write:
                return gridded.to_netcdf(
                    path, compute=False, encoding={proxy_cfg.name: comp}
                )
            else:
                return gridded

        shares_fname = (
            proxy_cfg.template.format(
                name=f"{proxy_cfg.name}-shares", **ids, **iter_ids
            ).replace(" ", "__")
            + ".nc"
        )
        shares_path = self.output_dir / shares_fname
        logger().info(f"Writing to {shares_path}")

        total = gridded.sum(share_dims)
        shares = gridded / total
        total = total.to_dataset(name=proxy_cfg.name)
        shares = shares.to_dataset(name=f"{proxy_cfg.name}-shares")
        if write:
            return total.to_netcdf(
                path, compute=False, encoding={proxy_cfg.name: comp}
            ), shares.to_netcdf(
                shares_path, compute=False, encoding={f"{proxy_cfg.name}-shares": comp}
            )
        else:
            return total, shares
