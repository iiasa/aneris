import numpy as np
import ptolemy as pt
import pycountry
import xarray as xr

from aneris import utils
from aneris.errors import MissingColumns, MissingCoordinateValue, MissingDimension


def country_name(country: str):
    return pycountry.countries.get(alpha_3=country).name


def check_coord_overlap(
    x, y, coord, x_strict=False, y_strict=False, warn=False, labels=None
):
    """
    Checks whether the coordinates or columns between two xarray.DataArrays.

    Parameters
    ----------
    x : xarray.DataArray
    y : xarray.DataArray
    coord : str
    x_strict : bool, optional
        the check fails if the coordinates in `y` are not a subset of `x`
    y_strict : bool, optional
        the check fails if the coordinates in `x` are not a subset of `y`
    warn : bool, optional
        if the check fails, issue a warning rather than a `MissingCoordinateValue` error
    labels : callable or dict
        what to report for missing coordinates

    Raises
    ------
    `MissingCoordinateValue` if check fails
    """
    if labels is None:
        labels = lambda x: x
    if isinstance(labels, dict):
        label_dict = labels
        labels = lambda x: label_dict.get(x, x)

    x, y = set(np.unique(x[coord])), set(np.unique(y[coord]))
    msg = ""
    if x_strict and x - y:
        missing = x - y
        msg += f"Missing from x {coord}: {', '.join(str(labels(x)) for x in missing)}\n"
    if y_strict and y - x:
        missing = y - x
        msg += f"Missing from y {coord}: {', '.join(str(labels(x)) for x in missing)}\n"
    if msg and not warn:
        raise MissingCoordinateValue(msg)
    elif msg and warn:
        utils.logger().warning(msg)


def grid(
    df,
    proxy,
    idx_raster,
    value_col="value",
    shape_col="iso",
    extra_coords=["year", "gas", "sector"],
    as_flux=False,
):
    """
    Develops spatial grids for emissions data.

    Parameters
    ----------
    df : pandas.DataFrame
        downscaled emissions provided per country (iso)
    proxy : xarray.DataArray
        proxy data used to apply emissions to spatial grids
    idx_raster : xarray.DataArray
        a raster mapping data in `df` to spatial grids
    value_col : str, optional
        the column in `df` which is gridded
    shape_col : str, optional
        the column in `df` which aligns with `idx_raster`
    extra_coords : Collection of str, optional
        the additional columns in `df` which will become coordinates
        in the returned DataArray
    as_flux : bool, optional
        if True, divide the result by the latitude-resolved cell areas
        to estimate parameter as a flux rather than bulk magnitude

    Returns
    -------
    xarray.DataArray:
        gridded emissions from `df`

    Notes
    -----
    1. `df` must have columns including `extra_coords`, `value_col`, and `shape_col`
    2. `proxy` must have coodrinates including `extra_coords`, "lat", and "lon"
    3. `idx_rater` must have coodrinates including `shape_col`, "lat", and "lon"
    """
    df_dim_diff = set(extra_coords + [value_col, shape_col]).difference(set(df.columns))
    if df_dim_diff:
        raise MissingColumns(f"df missing columns: {df_dim_diff}")
    proxy_dim_diff = set(extra_coords + ["lat", "lon"]).difference(set(proxy.dims))
    if proxy_dim_diff:
        raise MissingDimension(f"proxy missing dimensions: {proxy_dim_diff}")
    idxr_dim_diff = set([shape_col] + ["lat", "lon"]).difference(set(idx_raster.dims))
    if idxr_dim_diff:
        raise MissingDimension(f"idx_raster missing dimensions: {idxr_dim_diff}")

    map_data = pt.df_to_weighted_raster(
        df,
        xr.where(idx_raster > 0, 1, np.nan),
        col=value_col,
        extra_coords=extra_coords,
    )
    weighted_proxy = idx_raster * proxy
    normalized_proxy = weighted_proxy / weighted_proxy.sum(dim=["lat", "lon"])

    for coord in extra_coords:
        check_coord_overlap(
            normalized_proxy, map_data, coord, x_strict=True, y_strict=True
        )
    check_coord_overlap(normalized_proxy, map_data, shape_col, y_strict=True)
    # warn here because sometimes we have more small countries than data
    check_coord_overlap(normalized_proxy, map_data, shape_col, x_strict=True, warn=True)

    result = (map_data * normalized_proxy).sum(dim=shape_col)[value_col]
    if as_flux:
        lat_areas_in_m2 = xr.DataArray.from_series(pt.cell_area_from_file(proxy))
        result = result / lat_areas_in_m2
    return result
