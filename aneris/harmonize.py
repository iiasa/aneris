import os
import argparse
import warnings
import logging


import numpy as np
import pandas as pd


from aneris import utils
from aneris.methods import harmonize_factors, constant_offset, reduce_offset, \
    constant_ratio, reduce_ratio, linear_interpolate, model_zero, hist_zero, \
    coeff_of_var, default_methods


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


class HarmonizationDriver(object):

    def __init__(self, rc, model, hist, overrides, regions):
        self.prefix = rc['prefix']
        self.suffix = rc['suffix']
        self.config = rc['config']
        self.exog_files = rc['exogenous'] if 'exogenous' in rc else []
        self.model = model
        self.hist = hist
        self.overrides = overrides
        self.regions = regions

        model_names = self.model.Model.unique()
        if len(model_names) > 1:
            raise ValueError('Can not have more than one model to harmonize')
        self.model_name = model_names[0]

        self._aggregate_history()
        self.exogenous_trajectories = self._exogenous_trajectories()

        self._xlator = utils.FormatTranslator()

        self._model_dfs = []
        self._metadata_dfs = []

    def _exogenous_trajectories(self):
        # add exogenous variables
        dfs = []
        for fname in self.exog_files:
            exog = utils.pd_read(fname, sheetname='data')
            exog.columns = [str(x) for x in exog.columns]
            exog['Model'] = self.model_name
            exog['Scenario'] = ''
            cols = [c for c in model.columns if c in exog.columns]
            exog = exog[cols]
            dfs.append(exog)
        return pd.concat(dfs)

    def _aggregate_history(self):
        # check if global region exists, otherwise add it
        if not self.regions['ISO Code'].isin(['World']).any():
            glb = {
                'ISO Code': 'World',
                'Country': 'World',
                'Native Region Code': 'World',
            }
            print('Manually adding global regional definition: {}'.format(glb))
            self.regions = self.regions.append(glb, ignore_index=True)
        # aggregate historical to native regions
        print('Aggregating historical values to native regions')
        # must set verify to false for now because some isos aren't included!
        self.hist = utils.regionISOtoNative(
            self.hist, mapping=self.regions, verify=False)

    def _downselect_scen(self, model_name, scenario):
        ismodel = lambda df: df.Model == model_name
        isscen = lambda df: df.Scenario == scenario
        subset = lambda df: df[ismodel(df) & isscen(df)]

        self._model = subset(self._model)
        self._overrides = subset(self._overrides)

    def _downselect_var(self):
        # separate data
        print('Downselecting CEDS+|9+ variables')

        hasprefix = lambda df: df.Variable.str.startswith(self.prefix)
        hassuffix = lambda df: df.Variable.str.startswith(self.suffix)
        subset = lambda df: df[hasprefix(df) & hassuffix(df)]

        self._model = subset(self._model)
        self._hist = subset(self._hist)
        self._overrides = subset(self._overrides)

        if len(self._model) == 0:
            msg = 'No CEDS Variables found for harmonization. '
            'Searched for CEDS+|9+*|Unharmonized'
            raise ValueError(msg)
        assert(len(self._hist) > 0)

    def _to_std(self):
        print('Translating to standard format')
        self._model = (
            self.xlator.to_std(df=self._model.copy(), set_metadata=True)
            .set_index(utils.df_idx)
            .sort_index()
        )

        self._hist = (
            self._xlator.to_std(df=self._hist.copy(), set_metadata=False)
            .set_index(utils.df_idx)
            .sort_index()
        )
        # override CEDS with special cases (e.g. primap)
        self._hist = self._hist[~self._hist.index.duplicated(keep='last')]

        # hackery required because unit needed for df_idx
        if 'Unit' not in self._overrides:
            overrides['Unit'] = 'kt'
        self._overrides = (
            xlator.to_std(df=self._overrides.copy(), set_metadata=False)
            .set_index(utils.df_idx)
            .sort_index()
            .rename(str.lower)
        )['method']

    def _fill_model_trajectories(self):
            # add zeros to model values if not covered
        idx = self._hist.index
        notin = ~idx.isin(self._model.index)
        if notin.any():
            msg = 'Not all of self._history is covered by self._model: \n{}'
            _df = self._hist.loc[notin].reset_index()[utils.df_idx]
            warnings.warn(msg.format(_df.head()))
            zeros = pd.DataFrame(0, index=idx, columns=self._model.columns)
            self._model = self._model.combine_first(zeros)
        self._model = self._model.loc[idx]

    def _preprocess_trajectories(self):
        self._downselect_scen(model_name, scenario)  # only model and scen
        self._downselect_var()  # only prefix|*|suffix
        self._to_std()
        self._fill_model_trajectories()
        # TODO: check if overrides is empty!

    def _harmonize_global(self):
        self._glb_only, self._glb_meta = harmonize_global_total(
            self._glb_only, self._hist, self._overrides)

    def _harmonize_regional(self):
        # clean model
        self._model = subtract_regions_from_world(self._model, 'model')
        self._model = remove_recalculated_sectors(self._model)
        # remove rows with all 0s
        self._model = self._model[(self._model.T > 0).any()]

        # clean hist
        self._hist = subtract_regions_from_world(self._hist, 'hist')
        self._hist = remove_recalculated_sectors(self._hist)
        # remove rows with all 0s
        self._hist = self._hist[(self._hist.T > 0).any()]

    def _postprocess_trajectories(self):
        print('Translating to IAMC template')
        # update variable name
        self._model = self._model.reset_index()
        self._model.sector = self._model.sector.str.replace(
            self.suffix, 'Harmonized-DB')
        self._model = self._model.set_index(utils.df_idx)
        # from native to iamc format
        self._model = (
            xlator.to_template(self._model)
            .sort_index()
            .reset_index()
        )

        # add exogenous trajectories
        exog = self.exogenous_trajectories.copy()
        exog['Scenario'] = scenario
        self._model = pd.concat([self._model, exog])

    def harmonize(self, scenario):
        # need to specify model and scenario in xlator to template
        self._hist = self.hist.copy()
        self._model = self.model.copy()
        self._overrides = self.overrides.copy()

        self._preprocess_trajectories()
        self._split_global_only()
        self._harmonize_global()
        self._harmonize_regional()

        # combine special case results with harmonized results
        self._model = self._glb_only.combine_first(self._model)
        self._meta = self._glb_meta.combine_first(self._meta)

        # perform any automated diagnostics/analysis
        diagnostics(self._model, self._meta)

        # collect metadata
        self._meta = self._meta.reset_index()
        self._meta['model'] = self.model_name
        self._meta['scenario'] = scenario
        self._meta = self._meta.set_index(['model', 'scenario'])

        self._postprocess_trajectories()

        # store results
        self._model_dfs.append(self._model)
        self._metadata_dfs.append(self._meta)

    def scenarios(self):
        return self.model['Scenario'].unique()

    def harmonized_results(self):
        return (
            pd.concat(self._model_dfs),
            pd.concat(self._metadata_dfs),
        )


def harmonize_regional():

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
    if self.config['add_5regions']:
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
