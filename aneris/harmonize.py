import os
import argparse
import warnings

import numpy as np
import pandas as pd

import utils

_global_region = {
    'ISO Code': 'World',
    'Country': 'World',
    'Native Region Code': 'World',
}


def harmonize_factors(data, hist, harmonize_year='2015'):
    c, m = hist[harmonize_year], data[harmonize_year]
    offset = (c - m).fillna(0)
    offset.name = 'offset'
    ratios = (c / m).replace(np.inf, np.nan).fillna(0)
    ratios.name = 'ratio'
    return offset, ratios


def constant_offset(df, offset):
    df = df.copy()
    numcols = utils.numcols(df)
    # just add offset to all values
    df[numcols] = df[numcols].add(offset, axis=0)
    return df


def constant_ratio(df, ratios):
    df = df.copy()
    numcols = utils.numcols(df)
    # just add offset to all values
    df[numcols] = df[numcols].multiply(ratios, axis=0)
    return df


def linear_interpolate(df, offset, final_year='2050', harmonize_year='2015'):
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
    # current decision is to return a simple offset, this will be a straight
    # line for all time periods. previous behavior was to set df[numcols] = 0,
    # i.e., report 0 if model reports 0.
    return constant_offset(df, offset)


def hist_zero(df, *args, **kwargs):
    # TODO: should this set values to 0?
    df = df.copy()
    return df


def coeff_of_var(s):
    x = np.diff(s.values)
    return np.abs(np.std(x) / np.mean(x))


def default_methods(hist, model, base_year, luc_method=None):
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


def read_data(indfs):
    datakeys = sorted([x for x in indfs if x.startswith('data')])
    df = pd.concat([indfs[k] for k in datakeys])
    # don't know why reading from excel changes dtype and column types
    # but I have to reset them manually
    df.columns = df.columns.astype(str)
    numcols = [x for x in df.columns if x.startswith('2')]
    df[numcols] = df[numcols].astype(float)

    if '2015' not in df.columns:
        msg = 'Base year not found in model data. Existing columns are {}.'
        raise ValueError(msg.format(df.columns))

    # some teams also don't provide standardized column names and styles
    df.columns = df.columns.str.capitalize()

    return df


class Harmonizer(object):

    # WARNING: it is not possible to programmatically do the offset methods
    # because they use lambdas. you can't do `for y in years: lambda x: f(x,
    # kwarg=str(y))` because y is evaluated when the lambda is executed, not in
    # this block
    _methods = {
        'model_zero': model_zero,
        'hist_zero': hist_zero,
        'constant_ratio': constant_ratio,
        'constant_offset': constant_offset,
        'reduce_offset_2150_cov':
        lambda df, offsets: reduce_offset(df, offsets, final_year='2150'),
        'reduce_ratio_2150_cov':
        lambda df, ratios: reduce_ratio(df, ratios, final_year='2150'),
        'reduce_offset_2020':
        lambda df, offsets: reduce_offset(df, offsets, final_year='2020'),
        'reduce_offset_2040':
        lambda df, offsets: reduce_offset(df, offsets, final_year='2040'),
        'reduce_offset_2050':
        lambda df, offsets: reduce_offset(df, offsets, final_year='2050'),
        'reduce_offset_2080':
        lambda df, offsets: reduce_offset(df, offsets, final_year='2080'),
        'reduce_offset_2090':
        lambda df, offsets: reduce_offset(df, offsets, final_year='2090'),
        'reduce_offset_2100':
        lambda df, offsets: reduce_offset(df, offsets, final_year='2100'),
        'reduce_offset_2150':
        lambda df, offsets: reduce_offset(df, offsets, final_year='2150'),
        'reduce_ratio_2020':
        lambda df, ratios: reduce_ratio(df, ratios, final_year='2020'),
        'reduce_ratio_2050':
        lambda df, ratios: reduce_ratio(df, ratios, final_year='2050'),
        'reduce_ratio_2080':
        lambda df, ratios: reduce_ratio(df, ratios, final_year='2080'),
        'reduce_ratio_2090':
        lambda df, ratios: reduce_ratio(df, ratios, final_year='2090'),
        'reduce_ratio_2100':
        lambda df, ratios: reduce_ratio(df, ratios, final_year='2100'),
        'reduce_ratio_2150':
        lambda df, ratios: reduce_ratio(df, ratios, final_year='2150'),
        'linear_interpolate_2020':
        lambda df, offsets: linear_interpolate(df, offsets, final_year='2020'),
        'linear_interpolate_2050':
        lambda df, offsets: linear_interpolate(df, offsets, final_year='2050'),
        'linear_interpolate_2060':
        lambda df, offsets: linear_interpolate(df, offsets, final_year='2060'),
        'linear_interpolate_2080':
        lambda df, offsets: linear_interpolate(df, offsets, final_year='2080'),
        'linear_interpolate_2090':
        lambda df, offsets: linear_interpolate(df, offsets, final_year='2090'),
        'linear_interpolate_2100':
        lambda df, offsets: linear_interpolate(df, offsets, final_year='2100'),
        'linear_interpolate_2150':
        lambda df, offsets: linear_interpolate(df, offsets, final_year='2150'),

    }

    def __init__(self, data, history, config={}, verify_indicies=True):
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError('Data must use utils.df_idx')
        if not isinstance(history.index, pd.MultiIndex):
            raise ValueError('History must use utils.df_idx')
        if verify_indicies and not data.index.equals(history.index):
            idx = history.index.difference(data.index)
            msg = 'More history than model reports, adding 0 values {}'
            warnings.warn(msg.format(idx.to_series().head()))
            df = pd.DataFrame(0, columns=data.columns, index=idx)
            data = pd.concat([data, df]).sort_index().loc[history.index]
            assert data.index.equals(history.index)

        self.base_year = y = '2015'
        cols = [x for x in utils.numcols(data) if int(x) >= int(y)]
        self.data = data[cols]
        self.model = pd.Series(index=self.data.index, name='2015').to_frame()
        self.history = history
        self.methods_used = None
        self.offsets, self.ratios = harmonize_factors(self.data, self.history)

        key = 'default_luc_method'
        self.luc_method = config[key] if key in config else None

    def metadata(self):
        methods = self.methods_used
        if isinstance(methods, pd.Series):  # only defaults used
            methods = methods.to_frame()
            methods['default'] = methods['method']
            methods['override'] = ''

        meta = pd.concat([
            methods['method'],
            methods['default'],
            methods['override'],
            self.offsets,
            self.ratios,
            self.history[self.base_year],
            self.history.apply(coeff_of_var, axis=1),
            self.data[self.base_year],
            self.model[self.base_year],
        ], axis=1)
        meta.columns = [
            'method',
            'default',
            'override',
            'offset',
            'ratio',
            'history',
            'cov',
            'unharmonized',
            'harmonized',
        ]
        return meta

    def _default_methods(self):
        methods, diagnostics = default_methods(
            self.history, self.data, self.base_year, self.luc_method)
        return methods

    def _harmonize(self, method, idx, check_len):
        # get data
        model = self.data.loc[idx]
        hist = self.history.loc[idx]
        offsets = self.offsets.loc[idx]
        ratios = self.ratios.loc[idx]
        # get delta
        delta = ratios if 'ratio' in method else offsets

        # checks
        assert(not model.isnull().values.any())
        assert(not hist.isnull().values.any())
        assert(not delta.isnull().values.any())
        if check_len:
            assert((len(model) < len(self.data)) &
                   (len(hist) < len(self.history)))

        # harmonize
        model = Harmonizer._methods[method](model, delta)

        y = str(self.base_year)
        if model.isnull().values.any():
            msg = '{} method produced NaNs: {}, {}'
            where = model.isnull().any(axis=1)
            raise ValueError(msg.format(method,
                                        model.loc[where, y],
                                        delta.loc[where]))

        # construct the full df of history and future
        return model

    def methods(self, overrides=None):
        # get method listing
        methods = self._default_methods()
        if overrides is not None:
            midx = self.model.index
            oidx = overrides.index

            # remove duplicate values
            dup = oidx.duplicated(keep='last')
            if dup.any():
                msg = 'Removing duplicated override entries found: {}\n'
                warnings.warn(msg.format(overrides.loc[dup]))
                overrides = overrides.loc[~dup]

            # get subset of overrides which are in model
            outidx = oidx.difference(midx)
            if outidx.size > 0:
                msg = 'Removing override methods not in processed model output:\n{}'
                warnings.warn(msg.format(overrides.loc[outidx]))
                inidx = oidx.intersection(midx)
                overrides = overrides.loc[inidx]

            # overwrite defaults with overrides
            final_methods = overrides.combine_first(methods).to_frame()
            final_methods['default'] = methods
            final_methods['override'] = overrides
            methods = final_methods

        return methods

    def harmonize(self, overrides=None):
        # get special configurations
        methods = self.methods(overrides=overrides)

        # save for future inspection
        self.methods_used = methods
        if isinstance(methods, pd.DataFrame):
            methods = methods['method']  # drop default and override info
        if (methods == 'unicorn').any():
            msg = """Values found where model has positive and negative values
            and is zero in base year. Unsure how to proceed:\n{}\n{}"""
            cols = ['history', 'unharmonized']
            df1 = self.metadata().loc[methods == 'unicorn', cols]
            df2 = self.data.loc[methods == 'unicorn']
            raise ValueError(msg.format(df1.reset_index(), df2.reset_index()))

        dfs = []
        y = str(self.base_year)
        for method in methods.unique():
            print('Harmonizing with {}'.format(method))
            # get subset indicies
            idx = methods[methods == method].index
            check_len = len(methods.unique()) > 1
            # harmonize
            df = self._harmonize(method, idx, check_len)
            if method not in ['model_zero', 'hist_zero']:
                close = (df[y] - self.history.loc[df.index, y]).abs() < 1e-5
                if not close.all():
                    report = df[~close][y].reset_index()
                    msg = """Harmonization failed with method {} harmonized \
                    values != historical values. This is likely due to an \
                    override in the following variables:\n\n{}
                    """
                    raise ValueError(msg.format(method, report))
            dfs.append(df)

        df = pd.concat(dfs).sort_index()
        self.model = df
        return df


def check_null(df, name, fail=False):
    anynull = df.isnull().values.any()
    if fail:
        assert(not anynull)
    if anynull:
        msg = 'Null (missing) values found for {} indicies: \n{}'
        _df = df[df.isnull().any(axis=1)].reset_index()[utils.df_idx]
        warnings.warn(msg.format(name, _df))
        df.dropna(inplace=True)


def harmonize_global_total(model, hist, overrides):
    gases = utils.harmonize_total_gases
    idx = (pd.IndexSlice['World', gases, 'CEDS+|9+ Sectors|Unharmonized'],
           pd.IndexSlice[:])
    h = hist.loc[idx].copy()
    try:
        m = model.loc[idx].copy()
    except TypeError:
        warnings.warn('Non-CEDS gases not found in model')
        return None, None
    # catch empty dfs if no global toatls are overriden
    if overrides is None:
        o = None
    else:
        gases = overrides.index.get_level_values('gas').intersection(gases)
        try:
            gases = gases if len(gases) > 1 else gases[0]
        except IndexError:  # thrown if no harmonize_total_gases
            o = None
        idx = (pd.IndexSlice['World', gases, 'CEDS+|9+ Sectors|Unharmonized'],
               pd.IndexSlice[:])
        try:
            o = overrides.loc[idx].copy()
        except TypeError:  # thrown if gases not
            o = None

    check_null(m, 'model')
    check_null(h, 'hist', fail=True)
    harmonizer = Harmonizer(m, h)
    print('Harmonizing (with example methods):')
    print(harmonizer.methods(overrides=o).head())
    if o is not None:
        print('and override methods:')
        print(o.head())
        o.to_csv('o.csv')
    m = harmonizer.harmonize(overrides=o)
    check_null(m, 'model')

    metadata = harmonizer.metadata()
    return m, metadata


def harmonize_scenario(model, hist, regions, overrides, config, add_5region=True):
    model_name = model.Model.iloc[0]
    scenario = model.Scenario.iloc[0]

    # separate data
    print('Downselecting CEDS+|9+ variables')
    rows = lambda df: (
        (df.Variable.str.startswith('CEDS+|9+')) &
        (df.Variable.str.endswith('Unharmonized'))
    )
    model = model[rows(model)]
    if len(model) == 0:
        msg = 'No CEDS Variables found for harmonization. Searched for CEDS+|9+*|Unharmonized'
        raise ValueError(msg)
    hist = hist[rows(hist)]
    assert(len(hist) > 0)

    # translate data to calculation format
    xlator = utils.FormatTranslator()

    print('Translating to standard format')
    model = (
        xlator.to_std(df=model.copy(), set_metadata=True)
        .set_index(utils.df_idx)
        .sort_index()
    )

    hist = (
        xlator.to_std(df=hist.copy(), set_metadata=False)
        .set_index(utils.df_idx)
        .sort_index()
    )
    # override CEDS with special cases (e.g. primap)
    hist = hist[~hist.index.duplicated(keep='last')]

    if overrides is not None:
        overrides = overrides[rows(overrides)]
        # hackery required because unit needed for df_idx
        overrides['Unit'] = 'kt'
        overrides = (
            xlator.to_std(df=overrides.copy(), set_metadata=False)
            .set_index(utils.df_idx)
            .sort_index()
        )
        overrides.columns = overrides.columns.str.lower()
        overrides = overrides['method']

    # aggregate historical to native regions
    # check if global region exists, otherwise add it
    if not regions['ISO Code'].isin(['World']).any():
        print('Manually adding global regional definition: {}'.format(
            _global_region))
        regions = regions.append(_global_region, ignore_index=True)
    # must set verify to false for now because some isos aren't included!
    print('Aggregating historical values to native regions')
    hist = utils.regionISOtoNative(hist, mapping=regions, verify=False)

    # add zeros to model values if not covered
    idx = hist.index
    notin = ~idx.isin(model.index)
    if notin.any():
        msg = 'Not all of history is covered by model: \n{}'
        _df = hist.loc[notin].reset_index()[utils.df_idx]
        warnings.warn(msg.format(_df.head()))
        zeros = pd.DataFrame(0, index=idx, columns=model.columns)
        model = model.combine_first(zeros)
    model = model.loc[idx]

    # Harmonize special cases (for now this is N2O, which should be
    # harmonized only on global values)
    xtramodel, xtrameta = harmonize_global_total(model.copy(), hist, overrides)

    # make global only global (not global + sum of regions)
    def subtract_regions_from_world(df, name):
        check_null(df, name)
        if (df.loc['World']['2015'] == 0).all():
            # some models (gcam) are not reporting any values in World
            # without this, you get `0 - sum(other regions)`
            warnings.warn('Empty global region found in ' + name)
            return df

        # sum all rows where region == World
        total = utils.combine_rows(df, 'region', 'World', sumall=True,
                                   others=[], rowsonly=True)
        # sum all rows where region != World
        nonglb = utils.combine_rows(df, 'region', 'World', sumall=False,
                                    others=None, rowsonly=True)
        glb = total.subtract(nonglb, fill_value=0)
        # pick up some precision issues
        # TODO: this precision is large because I have seen model results
        # be reported with this large of difference due to round off and values
        # approaching 0
        glb[(glb / total).abs() < 5e-2] = 0.
        df = glb.combine_first(df)
        check_null(df, name)
        return df

    # remove sectoral totals which will need to be recalculated after
    # harmonization
    def remove_recalculated_sectors(df):
        df = df.reset_index()
        # TODO: THIS IS A HACK, CURRENT GASES DEFINITION ASSUME IAMC NAMES
        gases = df.gas.isin(utils.sector_gases + ['SO2', 'NOX'])
        sectors = df.sector.apply(lambda x: len(x.split('|')) == 3)
        keep = ~(gases & sectors)
        return df[keep].set_index(utils.df_idx)

    # clean model
    model = subtract_regions_from_world(model, 'model')
    model = remove_recalculated_sectors(model)
    model = model[(model.T > 0).any()]  # remove rows with all 0s

    # clean hist
    hist = subtract_regions_from_world(hist, 'hist')
    hist = remove_recalculated_sectors(hist)
    hist = hist[(hist.T > 0).any()]  # remove rows with all 0s

    # harmonize
    check_null(model, 'model')
    check_null(hist, 'hist', fail=True)
    harmonizer = Harmonizer(model, hist, config=config)
    print('Harmonizing (with example methods):')
    print(harmonizer.methods(overrides=overrides).head())

    if overrides is not None:
        print('and override methods:')
        print(overrides.head())
    model = harmonizer.harmonize(overrides=overrides)
    check_null(model, 'model')
    metadata = harmonizer.metadata()

    # add aggregate variables
    totals = 'CEDS+|9+ Sectors|Unharmonized'
    if model.index.get_level_values('sector').isin([totals]).any():
        msg = 'Removing sector aggregates. Recalculating with harmonized totals.'
        warnings.warn(msg)
        model.drop(totals, level='sector', inplace=True)
    model = (
        utils.EmissionsAggregator(model)
        .add_variables(totals=totals, aggregates=False)
        .df
        .set_index(utils.df_idx)
    )
    check_null(model, 'model')

    # combine regional values to send back into template form
    model.reset_index(inplace=True)
    model = model.set_index(utils.df_idx).sort_index()
    glb = utils.combine_rows(model, 'region', 'World',
                             sumall=True, rowsonly=True)
    model = glb.combine_first(model)

    # add 5regions
    if add_5region:
        print('Adding 5region values')
        # explicitly don't add World, it already exists from aggregation
        mapping = regions[regions['Native Region Code'] != 'World'].copy()
        model = model.append(utils.regionNativeto5(model, mapping=mapping))
        assert(not model.isnull().values.any())

    # duplicates come in from World and World being translated
    duplicates = model.index.duplicated(keep='first')
    if duplicates.any():
        regions = model[duplicates].index.get_level_values('region').unique()
        msg = 'Dropping duplicate rows found for regions: {}'.format(regions)
        warnings.warn(msg)
        model = model[~duplicates]

    # combine special case results with harmonized results
    model = xtramodel.combine_first(model)
    metadata = xtrameta.combine_first(metadata)

    # perform any automated diagnostics/analysis
    diagnostics(model, metadata)

    print('Translating to IAMC template')
    # update variable name
    model = model.reset_index()
    model.sector = model.sector.str.replace('Unharmonized', 'Harmonized-DB')
    model = model.set_index(utils.df_idx)
    # from native to iamc format
    model = xlator.to_template(model).sort_index().reset_index()

    # add exogenous variables
    f = os.path.join(utils.here, 'exogenous', 'ODS_future.xlsx')
    exog = utils.pd_read(f, sheetname='data')
    exog.columns = [str(x) for x in exog.columns]
    exog['Model'] = model_name
    exog['Scenario'] = scenario
    cols = [c for c in model.columns if c in exog.columns]
    exog = exog[cols]
    model = pd.concat([model, exog])

    # collect metadta
    metadata = metadata.reset_index()
    metadata['model'] = model_name
    metadata['scenario'] = scenario
    metadata = metadata.set_index(['model', 'scenario'])

    return model, metadata


def diagnostics(model, metadata):
    #
    # Detect Large Missing Values
    #
    num = metadata['history']
    denom = metadata['history'].groupby(level=['region', 'gas']).sum()

    # special merge because you can't do operations on multiindex
    ratio = pd.merge(num.reset_index(),
                     denom.reset_index(),
                     on=['region', 'gas'])
    ratio = ratio['history_x'] / ratio['history_y']
    ratio.index = num.index
    ratio.name = 'fraction'

    # downselect
    big = ratio[ratio > 0.2]
    bigmethods = metadata.loc[big.index, 'method']
    bad = bigmethods[bigmethods == 'model_zero']
    report = big.loc[bad.index].reset_index()

    if not report.empty:
        warnings.warn('LARGE MISSING Values Found!!:\n {}'.format(report))

    #
    # Detect non-negative CO2 emissions
    #
    m = model.reset_index()
    m = m[m.gas != 'CO2']
    neg = m[(m[utils.numcols(m)].T < 0).any()]

    if not neg.empty:
        warnings.warn(
            'Negative Emissions found for non-CO2 gases:\n {}'.format(neg))
        raise ValueError('Harmonization failed due to negative non-CO2 gases')


def main(inf, history=None, regions=None, output_prefix=None, add_5region=True):
    # default files
    history = history or utils.hist_path('history.csv')
    regions = regions or utils.region_path('message.csv')

    # read input
    hist = utils.pd_read(history)
    if 'github' in hist.columns[0]:
        raise ValueError('Git LFS file not updated.')

    regions = utils.pd_read(regions)
    indfs = utils.pd_read(inf, sheetname=None, encoding='utf-8')
    # make an empty df which will be caught later
    overrides = indfs['harmonization'] if 'harmonization' in indfs \
        else pd.DataFrame([], columns=['Scenario'])

    # get run control
    config = {}
    if'Configuration' in overrides:
        config = overrides[['Configuration', 'Value']].dropna()
        config = config.set_index('Configuration').to_dict()['Value']
        overrides = overrides.drop(['Configuration', 'Value'], axis=1)

    if regions.empty:
        raise ValueError('Region definition is empty')

    model = read_data(indfs)
    model.columns = model.columns.str.capitalize()
    model_name = model.Model.iloc[0]

    scenarios = model.Scenario.unique()
    dfs = []
    metadata = []
    for scenario in scenarios:
        print('Harmonizing {}'.format(scenario))
        _model = model[model.Scenario == scenario]
        _overrides = overrides[overrides.Scenario == scenario]
        _overrides = None if _overrides.empty else _overrides
        try:
            df, meta = harmonize_scenario(
                _model, hist, regions, _overrides, config, add_5region=add_5region)
            dfs.append(df)
            metadata.append(meta)
        except Exception as e:
            msg = 'Scenario {} failed with the following message:'
            print(msg.format(scenario))
            raise

    model = pd.concat(dfs)
    metadata = pd.concat(metadata)

    # write to excel
    prefix = output_prefix or inf.split('.')[0]
    fname = '{}_harmonized.xlsx'.format(prefix)
    print('Writing result to: {}'.format(fname))
    utils.pd_write(model, fname, sheet_name='data')

    # save data about harmonization
    fname = '{}_metadata.xlsx'.format(prefix)
    print('Writing metadata to: {}'.format(fname))
    utils.pd_write(metadata, fname)


if __name__ == '__main__':
    # C<LI
    descr = """
    Harmonize CEDS variables to data in the IAMC template format.

    Example usage:

    python harmonize.py input.xlsx history.csv --regions regions.csv
    """
    parser = argparse.ArgumentParser(description=descr,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    data_in = 'Input data file.'
    parser.add_argument('data_in', help=data_in)
    history = 'Historical emissions in the base year.'
    parser.add_argument('--history', help=history)
    regions = 'Mapping of country iso-codes to native regions.'
    parser.add_argument('--regions', help=regions)

    # parse cli
    args = parser.parse_args()
    inf = args.data_in
    history = args.history
    regions = args.regions or utils.region_path('message.csv')

    # run
    main(inf, history=history, regions=regions)
