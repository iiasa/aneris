import logging
import os
import re
from functools import reduce
from operator import and_

import numpy as np
import pandas as pd


# Index for iamc
iamc_idx = ["Model", "Scenario", "Region", "Variable"]

# default dataframe index
df_idx = ["region", "gas", "sector", "unit"]

# paths to data dependencies
here = os.path.join(os.path.dirname(os.path.realpath(__file__)))
hist_path = lambda f: os.path.join(here, "historical", f)
iamc_path = lambda f: os.path.join(here, "iamc_template", f)
region_path = lambda f: os.path.join(here, "regional_definitions", f)

# gases reported in kt of species
kt_gases = [
    "N2O",
    "SF6",
    "CF4",  # explicit species of PFC
    "C2F6",  # explicit species of PFC
    # individual f gases removed for now
    # # hfcs
    # 'HFC23', 'HFC32', 'HFC43-10', 'HFC125', 'HFC134a', 'HFC143a', 'HFC227ea', 'HFC245fa',
    # CFCs
    "CFC-11",
    "CFC-12",
    "CFC-113",
    "CFC-114",
    "CFC-115",
    "CH3CCl3",
    "CCl4",
    "HCFC-22",
    "HCFC-141b",
    "HCFC-142b",
    "Halon1211",
    "Halon1301",
    "Halon2402",
    "Halon1202",
    "CH3Br",
    "CH3Cl",
]

# gases reported in co2-equiv
co2_eq_gases = [
    "HFC",
]

# gases reported in Mt of species
mt_gases = [
    # IAMC names
    "BC",
    "CH4",
    "CO2",
    "CO",
    "NOx",
    "OC",
    "Sulfur",
    "NH3",
    "VOC",
    # non-IAMC names
    "SO2",
    "NOX",
    "NMVOC",
]

all_gases = sorted(kt_gases + co2_eq_gases + mt_gases)

# gases for which only sectoral totals are reported
total_gases = ["SF6", "CF4", "C2F6"] + co2_eq_gases

# gases for which only sectoral totals are harmonized
harmonize_total_gases = ["N2O"] + total_gases

# gases for which full sectoral breakdown is reported
sector_gases = sorted(set(all_gases) - set(total_gases))

# mapping for some gases whose names have changed recently
# TODO: can we remove this?
# TODO: should probably be a dictionary..
std_to_iamc_gases = [
    ("SO2", "Sulfur"),
    ("NOX", "NOx"),
    ("NMVOC", "VOC"),
]

# mapping from gas name to name to use in units
unit_gas_names = {
    "Sulfur": "SO2",
    "Kyoto Gases": "CO2-equiv",
    "F-Gases": "CO2-equiv",
    "HFC": "CO2-equiv",
    "PFC": "CO2-equiv",
    "CFC": "CO2-equiv",
}

_logger = None


def logger():
    """Global Logger used for aneris"""
    global _logger
    if _logger is None:
        logging.basicConfig()
        _logger = logging.getLogger()
        _logger.setLevel("INFO")
    return _logger


def isstr(x):
    """Returns True if x is a string"""
    try:
        return isinstance(x, (str, unicode))
    except NameError:
        return isinstance(x, str)


def isnum(s):
    """Returns True if s is a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def numcols(df):
    """Returns all columns in df that have data types of floats or ints"""
    dtypes = df.dtypes
    return [i for i in dtypes.index if dtypes.loc[i].name.startswith(("float", "int"))]


def check_null(df, name=None, fail=False):
    """Determines which values, if any in a dataframe are null

    Parameters
    ----------
    df : pd.DataFrame
    name : string, optional
        the name of the dataframe to use in a warning message
    fail : bool, optional
        if True, assert that no null values exist
    """
    anynull = df.isnull().values.any()
    if fail:
        assert not anynull
    if anynull:
        msg = "Null (missing) values found for {} indicies: \n{}"
        _df = df[df.isnull().any(axis=1)].reset_index()[df_idx]
        logger().warning(msg.format(name, _df))
        df.dropna(inplace=True, axis=1)


def gases(var_col):
    """The gas associated with each variable"""
    gasidx = lambda x: x.split("|").index("Emissions") + 1
    return var_col.apply(lambda x: x.split("|")[gasidx(x)])


def units(var_col):
    """returns a units column given a variable column"""
    gas_col = gases(var_col)

    # replace all gas names where name in unit != name in variable,
    # this can go away if we agree on the list
    replace = lambda x: x if x not in unit_gas_names else unit_gas_names[x]
    gas_col = gas_col.apply(replace)

    return gas_col.apply(
        lambda gas: "{} {}/yr".format("kt" if gas in kt_gases else "Mt", gas)
    )


def remove_emissions_prefix(x, gas="XXX"):
    """Return x with emissions prefix removed, e.g.,
    Emissions|XXX|foo|bar -> foo|bar
    """
    return re.sub(r"^Emissions\|{}\|".format(gas), "", x)



def isin(df=None, **filters):
    """Constructs a MultiIndex selector

    Usage
    -----
    > df.loc[isin(region="World", gas=["CO2", "N2O"])]
    or with explicit df to get boolean mask
    > isin(df, region="World", gas=["CO2", "N2O"])
    """

    def tester(df):
        tests = (df.index.isin(np.atleast_1d(v), level=k) for k, v in filters.items())
        return reduce(and_, tests, next(tests))

    return tester if df is None else tester(df)


def pd_read(f, str_cols=False, *args, **kwargs):
    """Try to read a file with pandas, supports CSV and XLSX

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
    """Try to write a file with pandas, supports CSV and XLSX"""
    # guess whether to use index, unless we're told otherwise
    index = kwargs.pop("index", isinstance(df.index, pd.MultiIndex))

    if f.endswith("csv"):
        df.to_csv(f, index=index, *args, **kwargs)
    else:
        writer = pd.ExcelWriter(f)
        df.to_excel(writer, index=index, *args, **kwargs)
        writer.save()
