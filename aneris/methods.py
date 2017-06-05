"""This module defines all possible functional forms of harmonization methods
and the default decision tree for choosing which method to use.

"""

import pandas as pd
import numpy as np

from aneris import utils


def harmonize_factors(df, hist, harmonize_year='2015'):
    """Calculate offset and ratio values between data and history

    Parameters
    ----------
    df : pd.DataFrame
        model data
    hist : pd.DataFrame
        historical data
    harmonize_year : string, optional
        column name of harmonization year 

    Returns
    -------
    offset : pd.Series
       offset (history - model)
    ratio : pd.Series
       ratio (history / model)
    """
    c, m = hist[harmonize_year], df[harmonize_year]
    offset = (c - m).fillna(0)
    offset.name = 'offset'
    ratios = (c / m).replace(np.inf, np.nan).fillna(0)
    ratios.name = 'ratio'
    return offset, ratios


def constant_offset(df, offset):
    """Calculate constant offset harmonized trajectory

    Parameters
    ----------
    df : pd.DataFrame
        model data
    offset : pd.DataFrame
        offset data

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    numcols = utils.numcols(df)
    # just add offset to all values
    df[numcols] = df[numcols].add(offset, axis=0)
    return df


def constant_ratio(df, ratios):
    """Calculate constant ratio harmonized trajectory

    Parameters
    ----------
    df : pd.DataFrame
        model data
    ratio : pd.DataFrame
        ratio data

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    numcols = utils.numcols(df)
    # just add offset to all values
    df[numcols] = df[numcols].multiply(ratios, axis=0)
    return df


def linear_interpolate(df, offset, final_year='2050', harmonize_year='2015'):
    """Calculate linearly interpolated convergence harmonized trajectory

    Parameters
    ----------
    df : pd.DataFrame
        model data
    offset : pd.DataFrame
        offset data
    final_year : string, optional
        column name of convergence year 
    harmonize_year : string, optional
        column name of harmonization year 

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    x1, x2 = harmonize_year, final_year
    y1, y2 = offset + df[x1], df[x2]
    m = (y2 - y1) / (float(x2) - float(x1))
    b = y1 - m * float(x1)

    cols = [x for x in utils.numcols(df) if int(x) < int(final_year)]
    for c in cols:
        df[c] = m * float(c) + b
    return df


def reduce_offset(df, offset, final_year='2050', harmonize_year='2015'):
    """Calculate offset convergence harmonized trajectory

    Parameters
    ----------
    df : pd.DataFrame
        model data
    offset : pd.DataFrame
        offset data
    final_year : string, optional
        column name of convergence year 
    harmonize_year : string, optional
        column name of harmonization year 

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    yi, yf = int(harmonize_year), int(final_year)
    numcols = utils.numcols(df)
    # get factors that reduce from 1 to 0; factors before base year are > 1
    f = lambda year: -(year - yi) / float(yf - yi) + 1
    factors = [f(int(year)) if year <= final_year else 0.0 for year in numcols]
    # add existing values to offset time series
    offsets = pd.DataFrame(np.outer(offset, factors),
                           columns=numcols, index=offset.index)
    df[numcols] = df[numcols] + offsets
    return df


def reduce_ratio(df, ratios, final_year='2050', harmonize_year='2015'):
    """Calculate ratio convergence harmonized trajectory

    Parameters
    ----------
    df : pd.DataFrame
        model data
    ratio : pd.DataFrame
        ratio data
    final_year : string, optional
        column name of convergence year 
    harmonize_year : string, optional
        column name of harmonization year 

    Returns
    -------
    df : pd.DataFrame
        harmonized trajectories
    """
    df = df.copy()
    yi, yf = int(harmonize_year), int(final_year)
    numcols = utils.numcols(df)
    # get factors that reduce from 1 to 0, but replace with 1s in years prior
    # to harmonization
    f = lambda year: -(year - yi) / float(yf - yi) + 1
    prefactors = [f(int(harmonize_year))
                  for year in numcols if year < harmonize_year]
    postfactors = [f(int(year)) if year <=
                   final_year else 0.0 for year in numcols if year >= harmonize_year]
    factors = prefactors + postfactors
    # multiply existing values by ratio time series
    ratios = pd.DataFrame(np.outer(ratios - 1, factors),
                          columns=numcols, index=ratios.index) + 1
    df[numcols] = df[numcols] * ratios
    return df


def model_zero(df, offset):
    """Returns result of aneris.methods.constant_offset()"""
    # current decision is to return a simple offset, this will be a straight
    # line for all time periods. previous behavior was to set df[numcols] = 0,
    # i.e., report 0 if model reports 0.
    return constant_offset(df, offset)


def hist_zero(df, *args, **kwargs):
    """Returns df (no change)"""
    # TODO: should this set values to 0?
    df = df.copy()
    return df


def coeff_of_var(s):
    """Returns coefficient of variation of a Series 

    .. math:: c_v = \\frac{\\sigma(s^{\\prime}(t))}{\\mu(s^{\\prime}(t))}

    Parameters
    ----------
    s : pd.Series
        timeseries

    Returns
    -------
    c_v : float
        coefficient of variation
    """
    x = np.diff(s.values)
    return np.abs(np.std(x) / np.mean(x))


def default_methods(hist, model, base_year, luc_method=None):
    """Determine default harmonization methods to use.

    See <WEBSITE> for a graphical description of the decision tree.

    Parameters
    ----------
    hist : pd.DataFrame
        historical data
    model : pd.DataFrame
        model data
    base_year : string, int
        column name of harmonization year 
    luc_method : string, optional
        method to use for high coefficient of variation

    Returns
    -------
    methods : pd.Series
       default harmonization methods
    metadata : pd.DataFrame
       metadata regarding why each method was chosen
    """
    luc_method = luc_method or 'reduce_offset_2150_cov'
    y = str(base_year)
    h = hist[y]
    m = model[y]
    dH = (h - m).abs() / h
    f = h / m
    dM = (model.max(axis=1) - model.min(axis=1)).abs() / model.max(axis=1)
    neg_m = (model < 0).any(axis=1)
    pos_m = (model > 0).any(axis=1)
    zero_m = (model == 0).all(axis=1)
    go_neg = ((model.min(axis=1) - h) < 0).any()
    cov = hist.apply(coeff_of_var, axis=1)

    # special override for co2
    # do this check for testing purposes
    if isinstance(model.index, pd.MultiIndex) and 'gas' in model.index.names:
        isco2 = model.reset_index().gas == 'CO2'
        isco2 = isco2.values
    else:
        isco2 = False

    df = pd.DataFrame({
        'dH': dH, 'f': f, 'dM': dM,
        'neg_m': neg_m, 'pos_m': pos_m,
        'zero_m': zero_m, 'go_neg': go_neg,
        'cov': cov, 'isco2': isco2,
        'h': h, 'm': m,
    })

    # for choice flow chart see
    # https://drive.google.com/drive/folders/0B6_Oqvcg8eP9QXVKX2lFVUJiZHc
    def choice(row):
        # special cases
        if row.h == 0:
            return 'hist_zero'
        if row.zero_m:
            return 'model_zero'
        if np.isinf(row.f) and row.neg_m and row.pos_m:
            # model == 0 in base year, and model goes negative
            # and positive
            return 'unicorn'  # this shouldn't exist!

        # model 0 in base year?
        if np.isclose(row.m, 0):
            # goes negative?
            if row.neg_m:
                return 'reduce_offset_2080'
            else:
                return 'constant_offset'
        else:
            # is this co2?
            if row['isco2']:
                return 'reduce_ratio_2080'
            # is cov big?
            if np.isfinite(row['cov']) and row['cov'] > 10:
                return luc_method
            else:
                # dH small?
                if row.dH < 0.5:
                    return 'reduce_ratio_2080'
                else:
                    # goes negative?
                    if row.neg_m:
                        return 'reduce_ratio_2100'
                    else:
                        return 'constant_ratio'

    ret = df.apply(choice, axis=1)
    ret.name = 'method'
    return ret, df
