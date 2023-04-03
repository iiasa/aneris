import numpy as np
import ptolemy as pt
import pycountry
import xarray as xr

from aneris import utils
from aneris.errors import MissingColumns, MissingCoordinateValue, MissingDimension


def check_coord_overlap(
    x, y, coord, x_strict=False, y_strict=False, strict=False, warn=False
):
    x, y = set(np.unique(x[coord])), set(np.unique(y[coord]))
    msg = ""
    if strict:
        x_strict, y_strict = True, True
    if x_strict and x - y:
        missing = x - y
        if coord == "iso":
            missing = [pycountry.countries.get(alpha_3=c).name for c in missing]
        msg += f"Missing from x {coord}: {missing}\n"
    if y_strict and y - x:
        missing = y - x
        if coord == "iso":
            missing = [pycountry.countries.get(alpha_3=c).name for c in missing]
        msg += f"Missing from y {coord}: {missing}\n"
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
    # TODO: add docstrings
    # Note that area normalization has been kept with `as_flux`, but other unit conversions need to happen outside
    # this function: kg_per_mt = 1e9, s_per_yr = 365 * 24 * 60 * 60
    # Otherwise, operates as currently in `prototype_gridding.ipynb`
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

    for coord in ["gas", "sector", "year"]:
        check_coord_overlap(normalized_proxy, map_data, coord, strict=True)
    # warn here because sometimes we have more small countries than data
    check_coord_overlap(normalized_proxy, map_data, "iso", x_strict=True, warn=True)
    check_coord_overlap(normalized_proxy, map_data, "iso", y_strict=True)

    result = (map_data * normalized_proxy).sum(dim="iso")[value_col]
    if as_flux:
        lat_areas_in_m2 = xr.DataArray.from_series(pt.cell_area_from_file(proxy))
        result /= lat_areas_in_m2
    return result
