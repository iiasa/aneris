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


@dask.delayed
def verify_global_values(aggregated, tabular, proxy_name, index, reltol=1e-4):
    tab_df = tabular.groupby(level=index).sum().unstack("year")
    grid_df = aggregated.to_series().groupby(level=index).sum().unstack("year")
    grid_df, tab_df = grid_df.align(tab_df, join="inner")

    reldiff = abs(grid_df - tab_df) / tab_df
    if (reldiff >= reltol).any(axis=None):
        logger().warning(
            f"Yearly global totals relative values between grids and global data for ({proxy_name}) not within {reltol}:\n"
            f"{reldiff}"
        )
    else:
        logger().info(
            f"Yearly global totals relative values between grids and global data for ({proxy_name}) within tolerance"
        )
    return reldiff


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
                        f"Proxy {proxy_cfg.path.stem} missing dimensions: "
                        + ", ".join(proxy_missing_dims)
                    )

                index = MultiIndex.from_product([get_index(dim) for dim in self.index])
                missing_from_data = index.difference(data_index)
                if not missing_from_data.empty:
                    msg = (
                        f"Proxy '{proxy_cfg.path.stem}' has values missing from `data`:\n"
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


    def _open_single_proxy(self, path, global_only=False, chunk_proxy_dims={}):
        opened = xr.open_dataarray(
            path,
            chunks=dict(zip(self.index, repeat(1))) | chunk_proxy_dims,
        )
        for idx in self.index:
            mapping = self.index_mappings.get(idx)
            if mapping is not None:
                opened[idx] = opened.indexes[idx].map(mapping)

        # TODO: this maybe isn't needed anymore with 'World' included in idxraster
        #       but need to confirm 'World' is also in the proxy rasters
        proxy = opened if global_only else self.idxraster * opened
        return {'to_close': opened, 'proxy': proxy}

    @contextmanager
    def open_and_normalize_proxy(self, cfgs, concat_dim='sector', as_flux=True, chunk_proxy_dims={}):  
        proxies = []
        to_close = []
        for _, cfg in cfgs.iterrows():
            _p = self._open_single_proxy(cfg['path'], cfg['global_only'], chunk_proxy_dims)
            proxies.append(_p['proxy'])
            to_close.append(_p['to_close'])

        try:
            proxy = xr.concat(proxies, dim=concat_dim) if len(proxies) > 1 else proxies[0]

            # NB: this only preserves seasonality if years and months are
            #     separate dimensions in the proxy raster. If instead they are
            #     combined into a single 'time' dimension, seasonality is lost.
            sum_spatial_dims = list(set(proxy.dims).intersection(self.spatial_dims))
            normalized = proxy / proxy.mean(self.mean_time_dims).sum(
                sum_spatial_dims
            )

            if as_flux:
                lat_areas_in_m2 = xr.DataArray.from_series(
                    pt.cell_area_from_file(proxy)
                )
                normalized = normalized / lat_areas_in_m2
            yield normalized

        finally:
            for f in to_close:
                f.close()

    def output_path(self, name, template, indexes, iter_ids, template_kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ids = {dim: index[0] for dim, index in indexes.items() if len(index) == 1}
        fname = (
            template.format(
                name=name.replace('_', '-'), **ids, **iter_ids, **template_kwargs
                ).replace(
                " ", "__"
            )
            + ".nc"
        )
        return self.output_dir / fname

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
        verify_output: bool = False,  # TODO: make docs
        skip_exists: bool = False, # TODO: make docs
        template_kwargs={},
        dress_up_callback = None,  # TODO: make docs
        encoding_kwargs = {}, # TODO: make docs
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
        for name, cfgs in self.proxy_cfg.groupby('name'):
            logger().info("Collecting tasks for proxy %s", name)
            def _get_unique_opt(cfgs, key):
                if len(cfgs[key].unique()) != 1:
                    raise ValueError(f'Non unique config keys {cfgs[key].unique()}')
                return cfgs[key].values[0]
            opts = {key: _get_unique_opt(cfgs, key) for key in ['template', 'as_flux', 'concat_dim']}
            
            with self.open_and_normalize_proxy(
                cfgs, 
                concat_dim=opts["concat_dim"], 
                as_flux=opts["as_flux"], 
                chunk_proxy_dims=chunk_proxy_dims,
                ) as proxy:
                write_tasks = []

                proxy_index = MultiIndex.from_product(
                    [proxy.indexes[dim] for dim in self.index]
                )
                # dropna is required when data is allowed to have less dimension
                # values than proxy (e.g., fewer years)
                tabular = semijoin(self.data, proxy_index, how="inner")

                for iter_vals in tabular.idx.unique(iter_levels):
                    iter_ids = dict(zip(iter_levels, iter_vals))
                    single_tabular = tabular.loc[isin(**iter_ids)].droplevel(
                        iter_levels
                    )
                    data = DataArray.from_series(single_tabular)

                    if skip_exists and self.output_path(name, opts["template"], data.indexes, iter_ids, template_kwargs).exists():
                        logger().info("File exists, skipping tasks for %s", iter_ids)
                        continue

                    logger().info("Adding tasks for %s", iter_ids)
                    gridded = (data * proxy).sum(self.country_level)

                    if verify_output:
                        write_tasks.append(
                            self.verify_output(name, single_tabular, gridded, as_flux=opts["as_flux"])
                        )
                    if write:
                        write_tasks.append(
                            self.compute_output(
                                name, 
                                opts["template"],
                                gridded,
                                data.indexes,
                                iter_ids,
                                template_kwargs=template_kwargs,
                                write=write,
                                callback=dress_up_callback,
                                encoding_kwargs=encoding_kwargs,
                            )
                        )

                with ProgressBar():
                    ret.append(dask.compute(write_tasks))

        return ret

    def verify_output(self, name, tabular, gridded, as_flux=True):
        # TODO: figure out correct message here
        # ids = {dim: index[0] for dim, index in indexes.items() if len(index) == 1}
        # logger().info(f"Veryifying output for {ids}")

        # TODO: this is complex and can be given to us by the user?
        # the point of this function is to compute global totals across
        # self.index (nominally sector, gas, year), and compare with
        # the same values summed up in the original tabular data provided
        # to confirm that gridded values comport with provided global totals
        sum_spatial_dims = list(set(gridded.dims).intersection(self.spatial_dims))

        if as_flux:
            lat_areas_in_m2 = xr.DataArray.from_series(pt.cell_area_from_file(gridded))
            gridded = gridded * lat_areas_in_m2

        aggregated = gridded.mean(dim=self.mean_time_dims).sum(dim=sum_spatial_dims)

        return verify_global_values(aggregated, tabular, name, self.index)

    # MJG: add a callback function here that dresses the dataset in input4MIPS style
    def compute_output(
        self,
        name,
        template,
        gridded: DataArray,
        indexes,
        iter_ids,
        write=True,
        callback=None,
        template_kwargs={},
        encoding_kwargs={},
    ):
        # TODO: need to add attr definitions and dimension bounds
        path = self.output_path(name, template, indexes, iter_ids, template_kwargs)
        logger().info(f"Writing to {path}")
        ids = {dim: index[0] for dim, index in indexes.items() if len(index) == 1}

        gridded = gridded.to_dataset(name=name)
        if callback:
            gridded = callback(gridded, **ids, **iter_ids, **template_kwargs)
        if write:
            return gridded.to_netcdf(
                path, compute=False, encoding={name: encoding_kwargs, 'time': dict(calendar='noleap')},
            )
        else:
            return gridded
