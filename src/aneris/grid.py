from contextlib import contextmanager
from itertools import repeat
from pathlib import Path
from typing import Optional, Sequence, Union

import dask
import pandas as pd
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
        """Prepare gridding data

        Parameters
        ----------
        data : DataFrame or Series
            Tabular data (should contain `country_level` and `index` levels)
            If a DataFrame is given, it expects 'year's on the columns.
        idxraster : DataArray
            Rasterized country map (should sum to 1 over `country_level`)
        proxy_cfg : DataFrame
            Configuration of proxies with the columns:
            "name", "path", "template", "as_flux", "separate_shares"
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
            if "separate_shares" not in proxy_cfg.columns:
                proxy_cfg["separate_shares"] = False
        self.proxy_cfg = proxy_cfg

        self.index = list(index)
        self.index_mappings = index_mappings if index_mappings is not None else dict()
        self.country_level = country_level

        self.output_dir = Path.cwd() if output_dir is None else Path(output_dir)

    def check(self) -> None:
        """Check levels and dimensions of gridding data

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
        countries_data = set(self.data.idx.unique(self.country_level))
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
                    raise MissingCoordinateValue(
                        f"Proxy '{proxy_cfg.name}' has values missing from `data`:\n"
                        + missing_from_data.to_frame().to_string(index=False)
                    )

                proxy_index.append(index)

        proxy_index = pd.concat(proxy_index)
        missing_from_proxy = data_index.difference(proxy_index)
        if not missing_from_proxy.empty:
            raise MissingCoordinateValue(
                "None of the configured proxies provides:\n"
                + missing_from_proxy.to_frame().to_string(index=False)
            )

    @contextmanager
    def open_and_normalize_proxy(self, proxy_cfg):
        with xr.open_dataarray(
            proxy_cfg.path, chunks=dict(zip(self.index, repeat(1)))
        ) as proxy:
            for idx in self.index:
                mapping = self.index_mappings.get(idx)
                if mapping is not None:
                    proxy[idx] = proxy.indexes[idx].map(mapping)

            separate = self.idxraster * proxy
            normalized = separate / separate.sum(["lat", "lon"])

            if proxy_cfg.as_flux:
                lat_areas_in_m2 = xr.DataArray.from_series(
                    pt.cell_area_from_file(proxy)
                )
                normalized = normalized / lat_areas_in_m2

            yield normalized

    def grid(self, skip_check: bool = False) -> None:
        """
        Grid data onto configured proxies

        Parameters
        ----------
        skip_check : bool, default False
            If set, skips structural and alignment checks
        """

        if not skip_check:
            self.check()

        iter_levels = self.data.index.names.difference(
            self.index + [self.country_level]
        )

        for proxy_cfg in self.proxy_cfg.itertuples():
            logger().info("Collecting tasks for proxy %s", proxy_cfg.name)

            with self.open_and_normalize_proxy(proxy_cfg) as proxy:
                write_tasks = []

                proxy_index = MultiIndex.from_product(
                    [proxy.indexes[dim] for dim in self.index]
                )
                tabular = semijoin(self.data, proxy_index, how="right")

                for iter_vals in tabular.idx.unique(iter_levels):
                    iter_ids = dict(zip(iter_levels, iter_vals))
                    logger().info("Adding tasks for %s", iter_ids)
                    data = DataArray.from_series(
                        tabular.loc[isin(**iter_ids)].droplevel(iter_levels)
                    )
                    gridded = (data * proxy).sum(self.country_level)

                    write_tasks.append(
                        self.write_output(proxy_cfg, gridded, data.indexes, iter_ids)
                    )

                with ProgressBar():
                    dask.compute(write_tasks)

    def write_output(self, proxy_cfg, gridded: DataArray, indexes, iter_ids):
        # TODO: need to add attr definitions and dimension bounds
        ids = {dim: index[0] for dim, index in indexes.items() if len(index) == 1}
        path = (
            self.output_dir
            / proxy_cfg.template.format(name=proxy_cfg.name, **ids, **iter_ids)
        ).with_suffix(".nc")
        logger().info(f"Writing to {path}")
        if not proxy_cfg.separate_shares:
            return gridded.to_dataset(name=proxy_cfg.name).to_netcdf(
                path, compute=False
            )

        if isinstance(proxy_cfg.separate_shares, str):
            shares_stem = proxy_cfg.separate_shares.format(name=proxy_cfg.name, **ids)
        else:
            shares_stem = proxy_cfg.template.format(
                name=f"{proxy_cfg.name}-shares", **ids
            )
        shares_path = (self.output_dir / shares_stem).with_suffix(".nc")

        shares_dims = [dim for dim in self.index if dim not in ids]
        total = gridded.sum(shares_dims)
        shares = gridded / total
        return total.to_dataset(name=proxy_cfg.name).to_netcdf(
            path, compute=False
        ), shares.to_dataset(name=f"{proxy_cfg.name}-shares").to_netcdf(
            shares_path, compute=False
        )
