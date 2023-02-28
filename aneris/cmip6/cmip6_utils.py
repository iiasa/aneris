import logging
import os
import re
from functools import reduce
from operator import and_

import numpy as np
import pandas as pd

from aneris.utils import (
    check_null,
    df_idx,
    gases,
    iamc_idx,
    iamc_path,
    isin, 
    kt_gases,
    logger,
    numcols,
    pd_read,
    region_path,
    remove_emissions_prefix,
    std_to_iamc_gases,
    units,
)

def recalculated_row_idx(df, prefix="", suffix=""):
    """Return a boolean array with rows that need to be recalculated.
    These are rows with total values for a gas species which is a sum of
    subsectors.
    During harmonization, subsector totals change, thus this summation must
    be recalculated.
    """
    df = df.reset_index()

    gas_sec_pairs = df[["gas", "sector"]].drop_duplicates()
    total_sector = "|".join([prefix, suffix])
    gases_with_subsectors = df.gas.isin(
        gas_sec_pairs[gas_sec_pairs.sector != total_sector].gas.unique()
    )
    is_sector_total = df.sector == total_sector
    return np.array(gases_with_subsectors & is_sector_total)


def remove_recalculated_sectors(df, prefix="", suffix=""):
    """Return df with Total gas (sum of all sectors) removed"""
    idx = recalculated_row_idx(df, prefix="", suffix="")
    return df[~idx]


def subtract_regions_from_world(df, name=None, base_year="2015", threshold=5e-2):
    """Subtract the sum of regional results in each variable from the World total.
    If the result is a World total below a threshold, set those values to 0.

    Parameters
    ----------
    df : pd.DataFrame
    name : string, optional
        name to use in error checking
    base_year : int, string, optional
        column to use in error checking
    threshold : float, optional
        threshold below which to set values to 0
    """
    # make global only global (not global + sum of regions)
    check_null(df, name)
    if (df.loc["World"][base_year] == 0).all():
        # some models (gcam) are not reporting any values in World
        # without this, you get `0 - sum(other regions)`
        logger().warning("Empty global region found in " + name)
        return df

    # sum all rows where region == World
    total = combine_rows(df, "region", "World", sumall=True, others=[], rowsonly=True)
    # sum all rows where region != World
    nonglb = combine_rows(
        df, "region", "World", sumall=False, others=None, rowsonly=True
    )
    glb = total.subtract(nonglb, fill_value=0)
    # pick up some precision issues
    # TODO: this precision is large because I have seen model results
    # be reported with this large of difference due to round off and values
    # approaching 0
    glb[(glb / total).abs() < threshold] = 0.0
    df = glb.combine_first(df)
    check_null(df, name)
    return df


def combine_rows(
    df,
    level,
    main,
    others=None,
    sumall=True,
    dropothers=True,
    rowsonly=False,
    newlabel=None,
):
    """Combine rows (add values) in a dataframe. Rows corresponding to the main and
    other values in a given level (or column) are added together and reattached
    taking the main value in the new column.

    For example, countries can be combined using this strategy.

    Parameters
    ----------
    df : pd.DataFrame
    level : string, int
        common level or column (e.g., 'region')
    main : string
        the value of the level to aggregate on
    others : string, optional
        a list of other values to aggregate
    sumall : bool, optional
        sum main and other values (otherwise, only add other values)
    dropothers : bool, optional
        remove rows with values provided in `others`
    rowsonly : bool, optional
        only return newly generated rows
    newlabel : string, optional
        a new label for the level/column value, default is main

    Returns
    -------
    df : pd.DataFrame
        resulting data
    """
    newlabel = newlabel or main
    multi_idx = isinstance(df.index, pd.MultiIndex)

    if multi_idx:
        df.reset_index(inplace=True)

    # get all values in level column
    lvl_values = df[level].unique()

    # if others is none, then its everything other than the primary
    others = others if others is not None else list(set(lvl_values) - set([main]))

    # set up df idx for operations
    grp_idx = [x for x in df_idx if x != level]
    df.set_index([level] + grp_idx, inplace=True)

    # generate new rows which are summation of subset of old rows
    sum_subset = [main] + others if sumall else others
    rows = df.loc[sum_subset].groupby(level=grp_idx).sum()
    rows[level] = newlabel
    rows = rows.set_index(level, append=True).reorder_levels(df_idx).sort_index()

    # get rid of rows that aren't needed in final dataframe
    drop = [main] + others if dropothers else [main]
    drop = list(set(drop) & set(lvl_values))
    df = df.drop(drop).reset_index().set_index(df_idx)

    # construct final dataframe
    df = rows if rowsonly else pd.concat([df, rows]).sort_index()

    if not multi_idx:
        df.reset_index(inplace=True)

    return df


def agg_regions(
    df, rfrom="ISO Code", rto="Native Region Code", mapping=None, verify=True
):
    """Aggregate values in a dataframe to a new regional composition

    Parameters
    ----------
    df : pd.DataFrame
    rfrom : string
        original regional composition column name in mapping
    rto : string
        column name to use for aggregation in mapping
    mapping : pd.DataFrame, optional
        mapping to use, otherwise MESSAGE mappings are read
    verify : bool, optional
        if True, confirm that sum of original values == sum of aggregated values

    Returns
    -------
    df : pd.DataFrame
    """
    mapping = (
        mapping if mapping is not None else pd.read_csv(region_path("message.csv"))
    )
    mapping[rfrom] = mapping[rfrom].str.upper()
    case_map = pd.Series(mapping[rto].unique(), index=mapping[rto].str.upper().unique())
    mapping[rto] = mapping[rto].str.upper()
    mapping = mapping[[rfrom, rto]].drop_duplicates().dropna()

    # unindex and set up values in correct form
    multi_idx = isinstance(df.index, pd.MultiIndex)
    if multi_idx:
        df = df.reset_index()
    df.region = df.region.str.upper()

    # remove regions without mappings
    check = mapping[rfrom]
    notin = list(set(df.region) - set(check))
    if len(notin) > 0:
        logger().warning("Removing regions without direct mapping: {}".format(notin))
        df = df[df.region.isin(check)]

    # map and sum
    dfto = (
        df.merge(mapping, left_on="region", right_on=rfrom, how="outer")
        .drop([rfrom, "region"], axis=1)
        .rename(columns={rto: "region"})
        .groupby(df_idx)
        .sum()
        .reset_index()
    )
    dfto.region = dfto.region.map(case_map)
    dfto = dfto.set_index(df_idx).sort_index()

    if verify:
        # contract on exit
        start = df[numcols(df)].values.sum()
        end = dfto[numcols(dfto)].values.sum()
        diff = abs(start - end)
        if np.isnan(diff) or diff / start > 1e-6:
            msg = "Difference between before and after is large: {}"
            raise (ValueError(msg.format(diff)))

    # revert form if needed
    if not multi_idx:
        dfto.reset_index(inplace=True)
    return dfto


class EmissionsAggregator(object):
    """Helper class to aggregate emissions"""

    def __init__(self, df, model=None, scenario=None):
        """Parameters
        ----------
        df : pd.DataFrame
            original data
        model : string, optional
            model name
        scenario : string, optional
            scenario name
        """
        self.multi_idx = isinstance(df.index, pd.MultiIndex)
        if self.multi_idx:
            df = df.reset_index()
        self.df = df
        self.model = model
        self.scenario = scenario
        assert (self.df.unit == "kt").all()

    def add_variables(self, totals=None, aggregates=True):
        """Add aggregates and variables with direct mappings.

        Parameters
        ----------
        totals : list, optional
             sectors to compute totals for
        add_aggregates : bool, optional
            whether to add aggregate variables
        """
        if totals is not None:
            self._add_totals(totals)
        if aggregates:
            self._add_aggregates()
        return self

    def to_template(self, **kwargs):
        """Create an IAMC template out of the original data frame

        Parameters
        ----------
        first_year: optional, the first year to report values for
        """
        self.df = FormatTranslator(self.df).to_template(
            model=self.model, scenario=self.scenario, **kwargs
        )
        return self.df

    def _add_totals(self, totals):
        assert not (self.df.sector == totals).any()
        grp_idx = [x for x in df_idx if x != "sector"]
        rows = self.df.groupby(grp_idx).sum().reset_index()
        rows["sector"] = totals
        self.df = pd.concat([self.df, rows])

    def _add_aggregates(self):
        mapping = pd_read(iamc_path("sector_mapping.xlsx"), sheet_name="Aggregates")
        mapping = mapping.applymap(remove_emissions_prefix)

        rows = []
        for sector in mapping["IAMC Parent"].unique():
            # mapping for aggregate sector for all gases
            _map = mapping[mapping["IAMC Parent"] == sector]
            _map = _map.set_index("IAMC Child")["IAMC Parent"]

            # rename variable column for subset of rows
            subset = self.df[self.df.sector.isin(_map.index)].copy()
            subset.sector = subset.sector.apply(lambda x: _map.loc[x])

            # add aggregate to rows
            subset = subset.groupby(df_idx).sum().reset_index()
            rows.append(subset)

        self.df = pd.concat([self.df] + rows)


class FormatTranslator(object):
    """Helper class to translate between IAMC and calcluation formats"""

    def __init__(self, df=None, prefix="", suffix=""):
        self.df = df if df is None else df.copy()
        self.model = None
        self.scenario = None
        self.prefix = prefix
        self.suffix = suffix

    def to_std(self, df=None, set_metadata=True, unit=True):
        """Translate a dataframe from IAMC to standard calculation format

        Parameters
        ----------
        df : pd.DataFrame, optional
        set_metadata : bool, optional
            save metadata (model, scenario) for future use
        unit : bool, optional
            check 'unit' col is present
        """
        df = self.df if df is None else df
        multi_idx = isinstance(df.index, pd.MultiIndex)
        if multi_idx:
            df.reset_index(inplace=True)

        if set(iamc_idx) - set(df.columns):
            msg = "Columns do not conform with IAMC index: {}"
            raise ValueError(msg.format(set(iamc_idx) - set(df.columns)))

        # make sure we're working with good data
        if len(df["Model"].unique()) > 1:
            raise ValueError("Model not unique: {}".format(df["Model"].unique()))
        assert len(df["Scenario"].unique()) <= 1
        assert df["Variable"].apply(lambda x: "Emissions" in x).all()

        # save data
        if set_metadata:
            self.model = df["Model"].iloc[0]
            self.scenario = df["Scenario"].iloc[0]

        # add std columns needed for conversions
        df["region"] = df["Region"]
        df["gas"] = gases(df["Variable"])
        df["sector"] = df["Variable"]
        if unit:
            df["unit"] = df["Unit"].apply(lambda x: x.split()[0])

        # convert gas names
        self._convert_gases(df, tostd=True)

        # convert units
        self._convert_units(df, tostd=True)

        # remove emissions prefix
        def update_sector(row):
            sectors = row.sector.split("|")
            idx = sectors.index("Emissions")
            sectors.pop(idx)  # emissions
            sectors.pop(idx)  # gas
            return "|".join(sectors).strip("|")

        if not df.empty:
            df["sector"] = df.apply(update_sector, axis=1)
        # drop old columns
        dropidx = iamc_idx.copy()
        if unit:
            dropidx += ["Unit"]
        df.drop(dropidx, axis=1, inplace=True)

        # set up index and column order
        df.set_index(df_idx, inplace=True)
        df.sort_index(inplace=True)

        if not multi_idx:
            df.reset_index(inplace=True)

        return df

    def to_template(self, df=None, model=None, scenario=None, column_style=None):
        """Translate a dataframe from standard calculation format to IAMC

        Parameters
        ----------
        df : pd.DataFrame, optional
        model : string, optional
            model name
        scenario : string, optional
            scenario name
        column_style : string
            column style (upper, lower, etc.) to use
        """
        df = self.df if df is None else df
        multi_idx = isinstance(df.index, pd.MultiIndex)
        if multi_idx:
            df.reset_index(inplace=True)
        model = model or self.model
        scenario = scenario or self.scenario

        if set(df.columns) != set(df_idx + numcols(df)):
            msg = "Columns do not conform with standard index: {}"
            raise ValueError(msg.format(df.columns))

        # convert gas names
        self._convert_gases(df, tostd=False)

        # convert units
        self._convert_units(df, tostd=False)

        # inject emissions prefix
        def update_sector(row):
            sectors = row.sector.split("|")
            idx = self.prefix.count("|") + 1
            sectors.insert(idx, "Emissions")
            sectors.insert(idx + 1, row.gas)
            return "|".join(sectors).strip("|")

        df["sector"] = df.apply(update_sector, axis=1)
        # write units correctly
        df["unit"] = units(df.sector)

        # add new columns, remove old
        df["Model"] = model
        df["Scenario"] = scenario
        df["Variable"] = df.sector
        df["Region"] = df.region
        df["Unit"] = df.unit
        df.drop(df_idx, axis=1, inplace=True)

        # unit magic to make it always first, would be easier if it was in idx.
        hold = df["Unit"]
        df.drop("Unit", axis=1, inplace=True)
        df.insert(0, "Unit", hold)

        # set up index and column order
        idx = iamc_idx
        if column_style == "upper":
            df.columns = df.columns.str.upper()
            idx = [x.upper() for x in idx]
        df.set_index(idx, inplace=True)
        df.sort_index(inplace=True)
        if not multi_idx:
            df.reset_index(inplace=True)

        return df

    def _convert_gases(self, df, tostd=True):
        # std to template
        convert = std_to_iamc_gases

        if tostd:  # template to std
            convert = [(t, s) for s, t in convert]

        # from, to
        for f, t in convert:
            for col in ["gas", "sector"]:
                df[col] = df[col].replace(f, t)

    def _convert_units(self, df, tostd=True):
        where = ~df.gas.isin(kt_gases)
        if tostd:
            df.loc[where, numcols(df)] *= 1e3
            df.loc[where, "unit"] = "kt"
            assert (df.unit == "kt").all()
        else:
            assert (df.unit == "kt").all()
            df.loc[where, numcols(df)] /= 1e3
            df.loc[where, "unit"] = "Mt"