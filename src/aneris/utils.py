import logging
import os
from pathlib import Path
from typing import TypeAlias

import pandas as pd
import pycountry


Pathy: TypeAlias = str | Path

_logger = None

# Index for iamc
iamc_idx = ["Model", "Scenario", "Region", "Variable"]

# default dataframe index
df_idx = ["region", "gas", "sector", "unit"]

# paths to data dependencies
here = os.path.join(os.path.dirname(os.path.realpath(__file__)))
hist_path = lambda f: os.path.join(here, "historical", f)
iamc_path = lambda f: os.path.join(here, "iamc_template", f)
region_path = lambda f: os.path.join(here, "regional_definitions", f)


def logger():
    """
    Global Logger used for aneris.
    """
    global _logger
    if _logger is None:
        logging.basicConfig()
        _logger = logging.getLogger()
        _logger.setLevel("INFO")
    return _logger


def numcols(df):
    """
    Returns all columns in df that have data types of floats or ints.
    """
    dtypes = df.dtypes
    return [i for i in dtypes.index if dtypes.loc[i].name.startswith(("float", "int"))]


def isstr(x):
    """
    Returns True if x is a string.
    """
    try:
        return isinstance(x, (str, unicode))
    except NameError:
        return isinstance(x, str)


def isnum(s):
    """
    Returns True if s is a number.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def pd_read(f, str_cols=False, *args, **kwargs):
    """
    Try to read a file with pandas, supports CSV and XLSX.

    Parameters
    ----------
    f : string
        the file to read in
    str_cols : bool, optional
        turn all columns into strings (numerical column names are sometimes
        read in as numerical dtypes)
    args, kwargs : sent directly to the Pandas read function

    Returns
    -------
    df : pd.DataFrame
    """
    if f.endswith("csv"):
        df = pd.read_csv(f, *args, **kwargs)
    else:
        df = pd.read_excel(f, *args, **kwargs)

    if str_cols:
        df.columns = [str(x) for x in df.columns]

    return df


def pd_write(df, f, *args, **kwargs):
    """
    Try to write a file with pandas, supports CSV and XLSX.
    """
    # guess whether to use index, unless we're told otherwise
    index = kwargs.pop("index", isinstance(df.index, pd.MultiIndex))

    if f.endswith("csv"):
        df.to_csv(f, index=index, *args, **kwargs)
    else:
        with pd.ExcelWriter(f) as writer:
            df.to_excel(writer, index=index, *args, **kwargs)


def normalize(s):
    return s / s.sum()


def country_name(iso: str):
    country_obj = pycountry.countries.get(alpha_3=iso)
    return iso if country_obj is None else country_obj.name


def skipempty(*dfs):
    return [df for df in dfs if not df.empty]
