import pandas as pd

from aneris import utils
from aneris import pd_read
from aneris.methods import harmonize_factors, constant_offset, reduce_offset, \
    constant_ratio, reduce_ratio, linear_interpolate, model_zero, hist_zero, \
    coeff_of_var, default_methods


def _log(msg, *args, **kwargs):
    utils.logger().info(msg, *args, **kwargs)


def _warn(msg, *args, **kwargs):
    utils.logger().warning(msg, *args, **kwargs)


class Harmonizer(object):
    """A class used to harmonize model data to historical data in the 
    standard calculation format
    """
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
        """Parameters
        ----------
        data : pd.DataFrame
            model data in standard calculation format
        history : pd.DataFrame
            history data in standard calculation format
        config : dict, optional
            configuration dictionary (see <WEBSITE> for options)
        verify_indicies : bool, optional
            check indicies of data and history, provide warning message if 
            different
        """
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError('Data must use utils.df_idx')
        if not isinstance(history.index, pd.MultiIndex):
            raise ValueError('History must use utils.df_idx')
        if verify_indicies and not data.index.equals(history.index):
            idx = history.index.difference(data.index)
            msg = 'More history than model reports, adding 0 values {}'
            _warn(msg.format(idx.to_series().head()))
            df = pd.DataFrame(0, columns=data.columns, index=idx)
            data = pd.concat([data, df]).sort_index().loc[history.index]
            assert data.index.equals(history.index)

        key = 'harmonize_year'
        # TODO type
        self.base_year = str(config[key]) if key in config else '2015'
        numcols = utils.numcols(data)
        cols = [x for x in numcols if int(x) >= int(self.base_year)]
        self.data = data[cols]
        self.model = pd.Series(index=self.data.index,
                               name=self.base_year).to_frame()
        self.history = history
        self.methods_used = None
        self.offsets, self.ratios = harmonize_factors(
            self.data, self.history, self.base_year)

        key = 'default_luc_method'
        self.luc_method = config[key] if key in config else None

    def metadata(self):
        """Return pd.DataFrame of method choice metadata"""
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
        """Return pd.DataFrame of methods to use for harmonization given 
        pd.DataFrame of overrides
        """
        # get method listing
        methods = self._default_methods()
        if overrides is not None:
            midx = self.model.index
            oidx = overrides.index

            # remove duplicate values
            dup = oidx.duplicated(keep='last')
            if dup.any():
                msg = 'Removing duplicated override entries found: {}\n'
                _warn(msg.format(overrides.loc[dup]))
                overrides = overrides.loc[~dup]

            # get subset of overrides which are in model
            outidx = oidx.difference(midx)
            if outidx.size > 0:
                msg = 'Removing override methods not in processed model output:\n{}'
                _warn(msg.format(overrides.loc[outidx]))
                inidx = oidx.intersection(midx)
                overrides = overrides.loc[inidx]

            # overwrite defaults with overrides
            final_methods = overrides.combine_first(methods).to_frame()
            final_methods['default'] = methods
            final_methods['override'] = overrides
            methods = final_methods

        return methods

    def harmonize(self, overrides=None):
        """Return pd.DataFrame of harmonized trajectories given pd.DataFrame 
        overrides
        """
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
            _log('Harmonizing with {}'.format(method))
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


class _TrajectoryPreprocessor(object):
    def __init__(self, hist, model, overrides, regions, prefix, suffix):
        self.hist = hist
        self.model = model
        self.overrides = overrides
        self.prefix = prefix
        self.suffix = suffix
        self.regions = regions

    def _downselect_scen(self, scenario):
        isscen = lambda df: df.Scenario == scenario
        self.model = self.model[isscen(self.model)]
        self.overrides = self.overrides[isscen(self.overrides)]

    def _downselect_var(self):
        # separate data
        select = '|'.join([self.prefix, self.suffix])
        _log('Downselecting {} variables'.format(select))

        hasprefix = lambda df: df.Variable.str.startswith(self.prefix)
        hassuffix = lambda df: df.Variable.str.endswith(self.suffix)
        subset = lambda df: df[hasprefix(df) & hassuffix(df)]

        self.model = subset(self.model)
        self.hist = subset(self.hist)
        self.overrides = subset(self.overrides)

        if len(self.model) == 0:
            msg = 'No Variables found for harmonization. Searched for {}.'
            raise ValueError(msg.format(select))
        assert(len(self.hist) > 0)

    def _to_std(self):
        _log('Translating to standard format')
        xlator = utils.FormatTranslator()

        self.model = (
            xlator.to_std(df=self.model.copy(), set_metadata=True)
            .set_index(utils.df_idx)
            .sort_index()
        )

        self.hist = (
            xlator.to_std(df=self.hist.copy(), set_metadata=False)
            .set_index(utils.df_idx)
            .sort_index()
        )
        # override with special cases if more are found in history
        self.hist = self.hist[~self.hist.index.duplicated(keep='last')]

        # hackery required because unit needed for df_idx
        if self.overrides.empty:
            self.overrides = None
        else:
            self.overrides['Unit'] = 'kt'
            self.overrides = (
                xlator.to_std(df=self.overrides.copy(), set_metadata=False)
                .set_index(utils.df_idx)
                .sort_index()
            )
            self.overrides.columns = self.overrides.columns.str.lower()
            self.overrides = self.overrides['method']

    def _agg_hist(self):
        # aggregate and clean hist
        _log('Aggregating historical values to native regions')
        # must set verify to false for now because some isos aren't included!
        self.hist = utils.agg_regions(
            self.hist, verify=False, mapping=self.regions,
            rfrom='ISO Code', rto='Native Region Code'
        )

    def _fill_model_trajectories(self):
            # add zeros to model values if not covered
        idx = self.hist.index
        notin = ~idx.isin(self.model.index)
        if notin.any():
            msg = 'Not all of self.history is covered by self.model: \n{}'
            _df = self.hist.loc[notin].reset_index()[utils.df_idx]
            _warn(msg.format(_df.head()))
            zeros = pd.DataFrame(0, index=idx, columns=self.model.columns)
            self.model = self.model.combine_first(zeros)
        self.model = self.model.loc[idx]

    def process(self, scenario):
        self._downselect_scen(scenario)  # only model and scen
        self._downselect_var()  # only prefix|*|suffix
        self._to_std()
        self._agg_hist()
        self._fill_model_trajectories()
        return self

    def results(self):
        return self.hist, self.model, self.overrides


class HarmonizationDriver(object):
    """A helper class to harmonize all scenarios for a model.
    """

    def __init__(self, rc, hist, model, overrides, regions):
        """Parameters
        ----------
        rc : aneris.RunControl
        hist : pd.DataFrame
            history in IAMC format
        model : pd.DataFrame
            model data in IAMC format
        overrides : pd.DataFrame
            harmonization overrides in IAMC format
        regions : pd.DataFrame
            regional aggregation mapping (ISO -> model regions)
        """
        self.prefix = rc['prefix']
        self.suffix = rc['suffix']
        self.config = rc['config']
        self.add_5regions = rc['add_5regions']
        self.exog_files = rc['exogenous'] if 'exogenous' in rc else []
        self.model = model
        self.hist = hist
        self.overrides = overrides
        self.regions = regions
        if not self.regions['ISO Code'].isin(['World']).any():
            glb = {
                'ISO Code': 'World',
                'Country': 'World',
                'Native Region Code': 'World',
            }
            _log('Manually adding global regional definition: {}'.format(glb))
            self.regions = self.regions.append(glb, ignore_index=True)

        model_names = self.model.Model.unique()
        if len(model_names) > 1:
            raise ValueError('Can not have more than one model to harmonize')
        self.model_name = model_names[0]
        self._xlator = utils.FormatTranslator(prefix=self.prefix,
                                              suffix=self.suffix)
        self._model_dfs = []
        self._metadata_dfs = []
        self.exogenous_trajectories = self._exogenous_trajectories()

        # TODO better type checking?
        self.config['harmonize_year'] = str(self.config['harmonize_year'])
        y = self.config['harmonize_year']
        if y not in model.columns:
            msg = 'Base year {} not found in model data. Existing columns are {}.'
            raise ValueError(msg.format(y, model.columns))
        if y not in hist.columns:
            msg = 'Base year {} not found in hist data. Existing columns are {}.'
            raise ValueError(msg.format(y, hist.columns))

    def _exogenous_trajectories(self):
        # add exogenous variables
        dfs = []
        for fname in self.exog_files:
            exog = pd_read(fname, sheetname='data')
            exog.columns = [str(x) for x in exog.columns]
            exog['Model'] = self.model_name
            dfs.append(exog)
        if len(dfs) == 0:  # add empty df if none were provided
            dfs.append(pd.DataFrame(columns=self.model.columns))
        return pd.concat(dfs)

    def _postprocess_trajectories(self, scenario):
        _log('Translating to IAMC template')
        # update variable name
        self._model = self._model.reset_index()
        self._model.sector = self._model.sector.str.replace(
            self.suffix, 'Harmonized-DB')
        self._model = self._model.set_index(utils.df_idx)
        # from native to iamc format
        self._model = (
            self._xlator.to_template(self._model, model=self.model_name,
                                     scenario=scenario)
            .sort_index()
            .reset_index()
        )

        # add exogenous trajectories
        exog = self.exogenous_trajectories.copy()
        if not exog.empty:
            exog['Scenario'] = scenario
        cols = [c for c in self._model.columns if c in exog.columns]
        exog = exog[cols]
        self._model = pd.concat([self._model, exog])

    def harmonize(self, scenario):
        """Harmonize a given scneario. Get results from 
        aneris.harmonize.HarmonizationDriver.results()
        """
        # need to specify model and scenario in xlator to template
        self._hist = self.hist.copy()
        self._model = self.model.copy()
        self._overrides = self.overrides.copy()

        # preprocess
        pp = _TrajectoryPreprocessor(self._hist, self._model, self._overrides,
                                     self.regions, self.prefix, self.suffix)
        # TODO, preprocess in init, just process here
        self._hist, self._model, self._overrides = pp.process(
            scenario).results()

        # global only gases
        self._glb_model, self._glb_meta = _harmonize_global_total(
            self.config, self.prefix, self.suffix,
            self._hist, self._model.copy(), self._overrides
        )

        # regional gases
        self._model, self._meta = _harmonize_regions(
            self.config, self.prefix, self.suffix, self.regions,
            self._hist, self._model.copy(), self._overrides,
            self.config['harmonize_year'], self.add_5regions
        )

        # combine special case results with harmonized results
        if self._glb_model is not None:
            self._model = self._glb_model.combine_first(self._model)
            self._meta = self._glb_meta.combine_first(self._meta)

        # perform any automated diagnostics/analysis
        diagnostics(self._model, self._meta)

        # collect metadata
        self._meta = self._meta.reset_index()
        self._meta['model'] = self.model_name
        self._meta['scenario'] = scenario
        self._meta = self._meta.set_index(['model', 'scenario'])
        self._postprocess_trajectories(scenario)

        # store results
        self._model_dfs.append(self._model)
        self._metadata_dfs.append(self._meta)

    def scenarios(self):
        """Return all known scenarios"""
        return self.model['Scenario'].unique()

    def harmonized_results(self):
        """Return 2-tuple of (pd.DataFrame of harmonized trajectories, 
        pd.DataFrame of metadata)
        """
        return (
            pd.concat(self._model_dfs),
            pd.concat(self._metadata_dfs),
        )


def _harmonize_global_total(config, prefix, suffix, hist, model, overrides):
    gases = utils.harmonize_total_gases
    sector = '|'.join([prefix, suffix])
    idx = (pd.IndexSlice['World', gases, sector],
           pd.IndexSlice[:])
    h = hist.loc[idx].copy()

    try:
        m = model.loc[idx].copy()
    except TypeError:
        _warn('Non-history gases not found in model')
        return None, None

    if m.empty:
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
        idx = (pd.IndexSlice['World', gases, sector],
               pd.IndexSlice[:])
        try:
            o = overrides.loc[idx].copy()
        except TypeError:  # thrown if gases not
            o = None

    utils.check_null(m, 'model')
    utils.check_null(h, 'hist', fail=True)
    harmonizer = Harmonizer(m, h, config=config)
    _log('Harmonizing (with example methods):')
    _log(harmonizer.methods(overrides=o).head())
    if o is not None:
        _log('and override methods:')
        _log(o.head())
    m = harmonizer.harmonize(overrides=o)
    utils.check_null(m, 'model')

    metadata = harmonizer.metadata()
    return m, metadata


def _harmonize_regions(config, prefix, suffix, regions, hist, model, overrides,
                       base_year, add_5regions):

    # clean model
    model = utils.subtract_regions_from_world(model, 'model', base_year)
    model = utils.remove_recalculated_sectors(model, prefix, suffix)
    # remove rows with all 0s
    model = model[(model.T > 0).any()]

    # clean hist
    hist = utils.subtract_regions_from_world(hist, 'hist', base_year)
    hist = utils.remove_recalculated_sectors(hist, prefix, suffix)
    # remove rows with all 0s
    hist = hist[(hist.T > 0).any()]

    if model.empty:
        raise RuntimeError(
            'Model is empty after downselecting regional values')

    # harmonize
    utils.check_null(model, 'model')
    utils.check_null(hist, 'hist', fail=True)
    harmonizer = Harmonizer(model, hist, config=config)
    _log('Harmonizing (with example methods):')
    _log(harmonizer.methods(overrides=overrides).head())

    if overrides is not None:
        _log('and override methods:')
        _log(overrides.head())
    model = harmonizer.harmonize(overrides=overrides)
    utils.check_null(model, 'model')
    metadata = harmonizer.metadata()

    # add aggregate variables
    totals = '|'.join([prefix, suffix])
    if model.index.get_level_values('sector').isin([totals]).any():
        msg = 'Removing sector aggregates. Recalculating with harmonized totals.'
        _warn(msg)
        model.drop(totals, level='sector', inplace=True)
    model = (
        utils.EmissionsAggregator(model)
        .add_variables(totals=totals, aggregates=False)
        .df
        .set_index(utils.df_idx)
    )
    utils.check_null(model, 'model')

    # combine regional values to send back into template form
    model.reset_index(inplace=True)
    model = model.set_index(utils.df_idx).sort_index()
    glb = utils.combine_rows(model, 'region', 'World',
                             sumall=True, rowsonly=True)
    model = glb.combine_first(model)

    # add 5regions
    if add_5regions:
        _log('Adding 5region values')
        # explicitly don't add World, it already exists from aggregation
        mapping = regions[regions['Native Region Code'] != 'World'].copy()
        aggdf = utils.agg_regions(model, mapping=mapping,
                                  rfrom='Native Region Code', rto='5_region')
        model = model.append(aggdf)
        assert(not model.isnull().values.any())

    # duplicates come in from World and World being translated
    duplicates = model.index.duplicated(keep='first')
    if duplicates.any():
        regions = model[duplicates].index.get_level_values('region').unique()
        msg = 'Dropping duplicate rows found for regions: {}'.format(regions)
        _warn(msg)
        model = model[~duplicates]

    return model, metadata


def diagnostics(model, metadata):
    """Provide warnings or throw errors based on harmonized model data and 
    metadata

    Current diagnostics are:
    - large missing values (sector has 20% or more contribution to 
      history and model does not report sector) 
      - Warning provided
    - non-negative CO2 emissions (values other than CO2 are < 0)
      - Error thrown

    Parameters
    ----------
    model : pd.DataFrame
        harmonized model data in standard calculation format
    metadata : pd.DataFrame
        harmonization metadata
    """
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
        _warn('LARGE MISSING Values Found!!:\n {}'.format(report))

    #
    # Detect non-negative CO2 emissions
    #
    m = model.reset_index()
    m = m[m.gas != 'CO2']
    neg = m[(m[utils.numcols(m)].T < 0).any()]

    if not neg.empty:
        _warn(
            'Negative Emissions found for non-CO2 gases:\n {}'.format(neg))
        raise ValueError('Harmonization failed due to negative non-CO2 gases')
