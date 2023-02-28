from __future__ import division

import numpy as np
import pandas as pd
from itertools import chain
from functools import partial
from pandas_indexing import projectlevel, semijoin

from aneris import utils
from aneris.utils import isin, pd_read
from aneris.methods import (
    harmonize_factors,
    constant_offset,
    reduce_offset,
    constant_ratio,
    reduce_ratio,
    linear_interpolate,
    model_zero,
    hist_zero,
    budget,
    coeff_of_var,
    default_methods,
)
from aneris.errors import (
    MissingHarmonisationYear,
    MissingHistoricalError,
    MissingScenarioError,
)

def _log(msg, *args, **kwargs):
    utils.logger().info(msg, *args, **kwargs)


def _warn(msg, *args, **kwargs):
    utils.logger().warning(msg, *args, **kwargs)


def _check_data(hist, scen, year, idx):
    # @coroa - this may be a very slow way to do this check..
    def downselect(df):
        return (
            df
            [year]
            .reset_index()
            .set_index(idx)
            .index
            .unique()
        )
    s = downselect(scen)
    h = downselect(hist)
    if h.empty:
        raise MissingHarmonisationYear(
            'No historical data in harmonization year'
        )

    if not s.difference(h).empty:
        raise MissingHistoricalError(
            'Historical data does not match scenario data in harmonization '
            f'year for\n {s.difference(h)}'
            )
    
    if not h.difference(s).empty:
        raise MissingScenarioError(
            'Scenario data does not match historical data in harmonization '
            f'year for\n {h.difference(s)}'
            )
    
def _check_overrides(overrides, idx):
    if overrides is None:
        return
    
    if not isinstance(overrides, pd.Series):
        raise TypeError('Overrides required to be pd.Series')
    
    if not overrides.name == 'method':
        raise ValueError('Overrides name must be method')
    
    if not overrides.index.name != idx:
        raise ValueError(f'Overrides must be indexed by {idx}')

class Harmonizer(object):
    """A class used to harmonize model data to historical data in the
    standard calculation format
    """

    _methods = {
        "model_zero": model_zero,
        "hist_zero": hist_zero,
        "budget": budget,
        "constant_ratio": constant_ratio,
        "constant_offset": constant_offset,
        "reduce_offset_2150_cov": partial(reduce_offset, final_year="2150"),
        "reduce_ratio_2150_cov": partial(reduce_ratio, final_year="2150"),
        **{
            f"{method.__name__}_{year}": partial(method, final_year=str(year))
            for year in chain(range(2020, 2101, 10), [2150])
            for method in (reduce_offset, reduce_ratio, linear_interpolate)
        },
    }

    def __init__(
        self, data, history, config={}, harm_idx=["region", "gas", "sector"], method_choice=None,
    ):
        """
        The Harmonizer class prepares and harmonizes historical data to
        model data.

        It has a strict requirement that all index values match between
        the historical and data DataFrames.


        Parameters
        ----------
        data : pd.DataFrame
            model data in standard calculation format
        history : pd.DataFrame
            history data in standard calculation format
        config : dict, optional
            configuration dictionary
            (see http://mattgidden.com/aneris/config.html for options)
        # TODO: add harm_index and method_choice
        """
        # check index consistency
        self.harm_idx = harm_idx
        data_check = projectlevel(data.index, harm_idx)
        hist_check = projectlevel(history.index, harm_idx)
        if not data_check.difference(hist_check).empty:
            raise ValueError(
                'Data to harmonize exceeds historical data avaiablility:\n'
                f'{data_check.difference(hist_check)}'
                )
        def check_idx(df, label):
            final_idx = harm_idx + ['unit']
            extra_idx = list(set(df.index.names) - set(final_idx))
            if extra_idx:
                df = df.droplevel(extra_idx)
                _warn(
                    f'Extra index found in {label}, dropping levels {extra_idx}'
                    )
            return df
        data = check_idx(data, 'data')
        history = check_idx(history, 'history')
        history.columns = history.columns.astype(data.columns.dtype)

        # set basic attributes
        self.data = data[utils.numcols(data)]
        self.history = history
        self.methods_used = None

        # set up defaults
        self.base_year = str(config["harmonize_year"]) if "harmonize_year" in config else None
        self.method_choice = method_choice

        # get default methods to use in decision tree
        self.ratio_method = config.get("default_ratio_method")
        self.offset_method = config.get("default_offset_method")
        self.luc_method = config.get("default_luc_method")
        self.luc_cov_threshold = config.get("luc_cov_threshold")

    def metadata(self):
        """Return pd.DataFrame of method choice metadata"""
        methods = self.methods_used
        if isinstance(methods, pd.Series):  # only defaults used
            methods = methods.to_frame()
            methods["default"] = methods["method"]
            methods["override"] = ""

        meta = pd.concat(
            [
                methods["method"],
                methods["default"],
                methods["override"],
                self.offsets,
                self.ratios,
                self.history[self.base_year],
                self.history.apply(coeff_of_var, axis=1),
                self.data[self.base_year],
                self.model[self.base_year],
            ],
            axis=1,
        )
        meta.columns = [
            "method",
            "default",
            "override",
            "offset",
            "ratio",
            "history",
            "cov",
            "unharmonized",
            "harmonized",
        ]
        return meta

    def _default_methods(self, year):
        assert year is not None
        methods, diagnostics = default_methods(
            self.history.droplevel(list(set(self.history.index.names) - set(self.harm_idx))),
            self.data.droplevel(list(set(self.data.index.names) - set(self.harm_idx))),
            year,
            method_choice=self.method_choice,
            ratio_method=self.ratio_method,
            offset_method=self.offset_method,
            luc_method=self.luc_method,
            luc_cov_threshold=self.luc_cov_threshold,
        )
        return methods

    def _harmonize(self, method, idx, check_len, base_year):
        # get data
        def downselect(df, idx, level='unit'):
            return df.reset_index(level=level).loc[idx].set_index(level, append=True)
        model = downselect(self.data, idx)
        hist = downselect(self.history, idx)
        offsets = downselect(self.offsets, idx)['offset']
        ratios = downselect(self.ratios, idx)['ratio']
        
        # get delta
        delta = hist if method == "budget" else ratios if "ratio" in method else offsets

        # checks
        assert not model.isnull().values.any()
        assert not hist.isnull().values.any()
        assert not delta.isnull().values.any()
        if check_len:
            assert (len(model) < len(self.data)) & (len(hist) < len(self.history))

        # harmonize
        model = Harmonizer._methods[method](model, delta, harmonize_year=base_year)

        y = str(base_year)
        if model.isnull().values.any():
            msg = "{} method produced NaNs: {}, {}"
            where = model.isnull().any(axis=1)
            raise ValueError(msg.format(method, model.loc[where, y], delta.loc[where]))

        # construct the full df of history and future
        return model

    def methods(self, year=None, overrides=None):
        # TODO: next issue is that other 'convenience' methods have less
        # robust override indices. need to decide how to support this
        """Return pd.DataFrame of methods to use for harmonization given
        pd.DataFrame of overrides
        """
        # get method listing
        base_year = year if year is not None else self.base_year or "2015"
        _check_overrides(overrides, self.harm_idx)
        methods = self._default_methods(year=base_year)

        if overrides is not None:
            # overrides requires an index
            if overrides.index.names == [None]:
                raise ValueError(
                    'overrides must have at least on index dimension '
                    f'aligned with methods: {methods.index.names}'
                    )
            # expand overrides index to match methods and align indicies
            overrides = (
                semijoin(overrides, methods.index, how="right")
                .reorder_levels(methods.index.names)
            )
            if not overrides.index.difference(methods.index).empty:
                raise ValueError(
                    'Data to override exceeds model data avaiablility:\n'
                    f'{overrides.index.difference(methods.index)}'
                    )
            overrides.name = methods.name
            
            # overwrite defaults with overrides
            final_methods = (
                overrides
                .combine_first(methods)
                .to_frame()
            )
            final_methods["default"] = methods
            final_methods["override"] = overrides
            methods = final_methods

        return methods


        

    def harmonize(self, year=None, overrides=None):
        """Return pd.DataFrame of harmonized trajectories given pd.DataFrame
        overrides
        """
        base_year = year if year is not None else self.base_year or "2015"
        _check_data(self.history, self.data, year, self.harm_idx)
        _check_overrides(overrides, self.harm_idx)

        self.model = pd.Series(
            index=self.data.index, name=base_year, dtype=float
        ).to_frame()
        self.offsets, self.ratios = harmonize_factors(
            self.data, self.history, base_year
        )
        # get special configurations
        methods = self.methods(year=year, overrides=overrides)

        # save for future inspection
        self.methods_used = methods
        if isinstance(methods, pd.DataFrame):
            methods = methods["method"]  # drop default and override info
        if (methods == "unicorn").any():
            msg = """Values found where model has positive and negative values
            and is zero in base year. Unsure how to proceed:\n{}\n{}"""
            cols = ["history", "unharmonized"]
            df1 = self.metadata().loc[methods == "unicorn", cols]
            df2 = self.data.loc[methods == "unicorn"]
            raise ValueError(msg.format(df1.reset_index(), df2.reset_index()))

        dfs = []
        y = base_year
        for method in methods.unique():
            _log("Harmonizing with {}".format(method))
            # get subset indicies
            idx = methods[methods == method].index
            check_len = len(methods.unique()) > 1
            # harmonize
            df = self._harmonize(method, idx, check_len, base_year=base_year)
            if method not in ["model_zero", "hist_zero"]:
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
        # only keep columns from base_year
        df = df[df.columns[df.columns.astype(int) >= int(base_year)]]
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
        select = "|".join([self.prefix, self.suffix])
        _log("Downselecting {} variables".format(select))

        hasprefix = lambda df: df.Variable.str.startswith(self.prefix)
        hassuffix = lambda df: df.Variable.str.endswith(self.suffix)
        subset = lambda df: df[hasprefix(df) & hassuffix(df)]

        self.model = subset(self.model)
        self.hist = subset(self.hist)
        self.overrides = subset(self.overrides)

        if len(self.model) == 0:
            msg = "No Variables found for harmonization. Searched for {}."
            raise ValueError(msg.format(select))
        assert len(self.hist) > 0

    def _to_std(self):
        _log("Translating to standard format")
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
        self.hist = self.hist[~self.hist.index.duplicated(keep="last")]

        # hackery required because unit needed for df_idx
        if self.overrides.empty:
            self.overrides = None
        else:
            self.overrides["Unit"] = "kt"
            self.overrides = (
                xlator.to_std(df=self.overrides.copy(), set_metadata=False)
                .set_index(utils.df_idx)
                .sort_index()
            )
            self.overrides.columns = self.overrides.columns.str.lower()
            self.overrides = self.overrides["method"]

    def _agg_hist(self):
        # aggregate and clean hist
        _log("Aggregating historical values to native regions")
        # must set verify to false for now because some isos aren't included!
        self.hist = utils.agg_regions(
            self.hist,
            verify=False,
            mapping=self.regions,
            rfrom="ISO Code",
            rto="Native Region Code",
        )

    def _fill_model_trajectories(self):
        # add zeros to model values if not covered
        idx = self.hist.index
        notin = ~idx.isin(self.model.index)
        if notin.any():
            msg = "Not all of self.history is covered by self.model: \n{}"
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
    """A helper class to harmonize all scenarios for a model."""

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
        self.prefix = rc["prefix"]
        self.suffix = rc["suffix"]
        self.config = rc["config"]
        self.add_5regions = rc["add_5regions"]
        self.exog_files = rc["exogenous"] if "exogenous" in rc else []
        self.model = model
        self.hist = hist
        self.overrides = overrides

        self.regions = regions
        if not self.regions["ISO Code"].isin(["World"]).any():
            glb = {
                "ISO Code": "World",
                "Country": "World",
                "Native Region Code": "World",
            }
            _log("Manually adding global regional definition: {}".format(glb))
            self.regions = self.regions.append(glb, ignore_index=True)

        model_names = self.model.Model.unique()
        if len(model_names) > 1:
            raise ValueError("Can not have more than one model to harmonize")
        self.model_name = model_names[0]
        self._xlator = utils.FormatTranslator(prefix=self.prefix, suffix=self.suffix)
        self._model_dfs = []
        self._metadata_dfs = []
        self._diagnostic_dfs = []
        self.exogenous_trajectories = self._exogenous_trajectories()

        # TODO better type checking?
        self.config["harmonize_year"] = str(self.config["harmonize_year"])
        y = self.config["harmonize_year"]
        if y not in model.columns:
            msg = "Base year {} not found in model data. Existing columns are {}."
            raise ValueError(msg.format(y, model.columns))
        if y not in hist.columns:
            msg = "Base year {} not found in hist data. Existing columns are {}."
            raise ValueError(msg.format(y, hist.columns))

    def _exogenous_trajectories(self):
        # add exogenous variables
        dfs = []
        for fname in self.exog_files:
            exog = pd_read(fname, sheet_name="data")
            exog.columns = [str(x) for x in exog.columns]
            exog["Model"] = self.model_name
            dfs.append(exog)
        if len(dfs) == 0:  # add empty df if none were provided
            dfs.append(pd.DataFrame(columns=self.model.columns))
        return pd.concat(dfs)

    def _postprocess_trajectories(self, scenario):
        _log("Translating to IAMC template")
        # update variable name
        self._model = self._model.reset_index()
        self._model.sector = self._model.sector.str.replace(
            self.suffix, self.config["replace_suffix"]
        )
        self._model = self._model.set_index(utils.df_idx)
        # from native to iamc format
        self._model = (
            self._xlator.to_template(
                self._model, model=self.model_name, scenario=scenario
            )
            .sort_index()
            .reset_index()
        )

        # add exogenous trajectories
        exog = self.exogenous_trajectories.copy()
        if not exog.empty:
            exog["Scenario"] = scenario
        cols = [c for c in self._model.columns if c in exog.columns]
        exog = exog[cols]
        self._model = pd.concat([self._model, exog])

    def harmonize(self, scenario, diagnostic_config=None):
        """Harmonize a given scneario. Get results from
        aneris.harmonize.HarmonizationDriver.results()

        Parameters
        ----------
        scenario : string
            a scenario name listed in the model data
        diagnostic_conifg: dictionary, optional
            configuration for use in the aneris.diagnostics() function
        """
        # need to specify model and scenario in xlator to template
        self._hist = self.hist.copy()
        self._model = self.model.copy()
        self._overrides = self.overrides.copy()
        self._regions = self.regions.copy()

        # preprocess
        pp = _TrajectoryPreprocessor(
            self._hist,
            self._model,
            self._overrides,
            self._regions,
            self.prefix,
            self.suffix,
        )
        # TODO, preprocess in init, just process here
        self._hist, self._model, self._overrides = pp.process(scenario).results()

        unharmonized = self._model.copy()

        # flag if this run will be with only global trajectories. if so, then
        # only global totals are harmonized, rest is skipped.
        global_harmonization_only = self.config["global_harmonization_only"]

        # global only gases
        self._glb_model, self._glb_meta = _harmonize_global_total(
            self.config,
            self.prefix,
            self.suffix,
            self._hist,
            self._model.copy(),
            self._overrides,
            default_global_gases=not global_harmonization_only,
        )

        if global_harmonization_only:
            self._model = self._glb_model
            self._meta = self._glb_meta
        else:
            # regional gases
            self._model, self._meta = _harmonize_regions(
                self.config,
                self.prefix,
                self.suffix,
                self._regions,
                self._hist,
                self._model.copy(),
                self._overrides,
                self.config["harmonize_year"],
                self.add_5regions,
            )

            # combine special case results with harmonized results
            if self._glb_model is not None:
                self._model = self._glb_model.combine_first(self._model)
                self._meta = self._glb_meta.combine_first(self._meta)

        # perform any automated diagnostics/analysis
        self._diag = diagnostics(
            unharmonized, self._model, self._meta, config=diagnostic_config
        )

        # collect metadata
        self._meta = self._meta.reset_index()
        self._meta["model"] = self.model_name
        self._meta["scenario"] = scenario
        self._meta = self._meta.set_index(["model", "scenario"])
        self._postprocess_trajectories(scenario)

        # store results
        self._model_dfs.append(self._model)
        self._metadata_dfs.append(self._meta)
        self._diagnostic_dfs.append(self._diag)

    def scenarios(self):
        """Return all known scenarios"""
        return self.model["Scenario"].unique()

    def harmonized_results(self):
        """Return 3-tuple of (pd.DataFrame of harmonized trajectories,
        pd.DataFrame of metadata, and similar of diagnostic information)
        """
        return (
            pd.concat(self._model_dfs),
            pd.concat(self._metadata_dfs),
            pd.concat(self._diagnostic_dfs),
        )


def _get_global_overrides(overrides, gases, sector):
    # None if no overlap with gases
    if overrides is None:
        return None

    # Downselect overrides that match global gas values
    o = overrides.loc[isin(region="World", sector=sector, gas=gases)]
    return o if not o.empty else None


def _harmonize_global_total(
    config, prefix, suffix, hist, model, overrides, default_global_gases=True
):
    all_gases = list(model.index.get_level_values("gas").unique())
    gases = utils.harmonize_total_gases if default_global_gases else all_gases
    sector = "|".join([prefix, suffix])
    idx = isin(region="World", gas=gases, sector=sector)
    h = hist.loc[idx].copy()

    try:
        m = model.loc[idx].copy()
    except TypeError:
        _warn("Non-history gases not found in model")
        return None, None

    if m.empty:
        return None, None

    # match override methods with global gases, None if no match
    o = _get_global_overrides(overrides, gases, sector)

    utils.check_null(m, "model")
    utils.check_null(h, "hist", fail=True)
    harmonizer = Harmonizer(m, h, config=config)
    _log("Harmonizing (with example methods):")
    _log(harmonizer.methods(year=harmonizer.base_year, overrides=o).head())
    if o is not None:
        _log("and override methods:")
        _log(o.head())
    m = harmonizer.harmonize(year=harmonizer.base_year, overrides=o)
    utils.check_null(m, "model")

    metadata = harmonizer.metadata()
    return m, metadata


def _harmonize_regions(
    config, prefix, suffix, regions, hist, model, overrides, base_year, add_5regions
):

    # clean model
    model = utils.subtract_regions_from_world(model, "model", base_year)
    model = utils.remove_recalculated_sectors(model, prefix, suffix)
    # remove rows with all 0s
    model = model[(model.T > 0).any()]

    # clean hist
    hist = utils.subtract_regions_from_world(hist, "hist", base_year)
    hist = utils.remove_recalculated_sectors(hist, prefix, suffix)

    # remove rows with all 0s
    hist = hist[(hist.T > 0).any()]

    if model.empty:
        raise RuntimeError("Model is empty after downselecting regional values")

    # harmonize
    utils.check_null(model, "model")
    utils.check_null(hist, "hist", fail=True)
    harmonizer = Harmonizer(model, hist, config=config)
    _log("Harmonizing (with example methods):")
    _log(harmonizer.methods(overrides=overrides).head())

    if overrides is not None:
        _log("and override methods:")
        _log(overrides.head())
    model = harmonizer.harmonize(overrides=overrides)
    utils.check_null(model, "model")
    metadata = harmonizer.metadata()

    # add aggregate variables. this works in three steps:
    # step 1: remove any sector total trajectories that also have subsectors to
    # be recalculated
    idx = utils.recalculated_row_idx(model, prefix, suffix)
    if idx.any():
        msg = "Removing sector aggregates. Recalculating with harmonized totals."
        _warn(msg)
        model = model[~idx]
    totals = "|".join([prefix, suffix])
    sector_total_idx = isin(model, sector=totals)
    subsector_idx = ~sector_total_idx
    # step 2: on the "clean" df, recalculate those totals
    subsectors_with_total_df = (
        utils.EmissionsAggregator(model[subsector_idx])
        .add_variables(totals=totals, aggregates=False)
        .df.set_index(utils.df_idx)
    )
    # step 3: recombine with model data that was sector total only
    sector_total_df = model[sector_total_idx]
    model = pd.concat([sector_total_df, subsectors_with_total_df])
    utils.check_null(model, "model")

    # combine regional values to send back into template form
    model.reset_index(inplace=True)
    model = model.set_index(utils.df_idx).sort_index()
    glb = utils.combine_rows(model, "region", "World", sumall=False, rowsonly=True)
    model = glb.combine_first(model)

    # add 5regions
    if add_5regions:
        _log("Adding 5region values")
        # explicitly don't add World, it already exists from aggregation
        mapping = regions[regions["Native Region Code"] != "World"].copy()
        aggdf = utils.agg_regions(
            model, mapping=mapping, rfrom="Native Region Code", rto="5_region"
        )
        model = pd.concat([model, aggdf])
        assert not model.isnull().values.any()

    # duplicates come in from World and World being translated
    duplicates = model.index.duplicated(keep="first")
    if duplicates.any():
        regions = model[duplicates].index.get_level_values("region").unique()
        msg = "Dropping duplicate rows found for regions: {}".format(regions)
        _warn(msg)
        model = model[~duplicates]

    return model, metadata


def diagnostics(unharmonized, model, metadata, config=None):
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
    unharmonized : pd.DataFrame
        unharmonized model data in standard calculation format
    model : pd.DataFrame
        harmonized model data in standard calculation format
    metadata : pd.DataFrame
        harmonization metadata
    config : dictionary, optional
        ratio values to use in diagnostics, key options include 'mid' and 'end'.
    """
    config = config or {"mid": 4.0, "end": 2.0}

    #
    # Detect Large Missing Values
    #
    num = metadata["history"]
    denom = metadata["history"].groupby(level=["region", "gas"]).sum()

    # special merge because you can't do operations on multiindex
    ratio = pd.merge(num.reset_index(), denom.reset_index(), on=["region", "gas"])
    ratio = ratio["history_x"] / ratio["history_y"]
    ratio.index = num.index
    ratio.name = "fraction"

    # downselect
    big = ratio[ratio > 0.2]
    bigmethods = metadata.loc[big.index, "method"]
    bad = bigmethods[bigmethods == "model_zero"]
    report = big.loc[bad.index].reset_index()

    if not report.empty:
        _warn("LARGE MISSING Values Found!!:\n {}".format(report))

    #
    # report on large medium an dlong-term differences
    #
    cols = utils.numcols(model)
    report = model.copy()
    mid, end = cols[len(cols) // 2 - 1], cols[-1]

    if "mid" in config:
        bigmid = np.abs(model[mid] - unharmonized[mid]) / unharmonized[mid]
        bigmid = bigmid[bigmid > config["mid"]]
        report["{}_diff".format(mid)] = bigmid

    if "end" in config:
        bigend = np.abs(model[end] - unharmonized[end]) / unharmonized[end]
        bigend = bigend[bigend > config["end"]]
        report["{}_diff".format(end)] = bigend

    report = report.drop(cols, axis=1).dropna(how="all")
    idx = metadata.index.intersection(report.index)
    report["method"] = metadata.loc[idx, "method"]
    report = report[~report["method"].isin(["model_zero", np.nan])]

    #
    # Detect non-negative CO2 emissions
    #
    m = model.reset_index()
    m = m[m.gas != "CO2"]
    neg = m[(m[utils.numcols(m)].T < 0).any()]

    if not neg.empty:
        _warn("Negative Emissions found for non-CO2 gases:\n {}".format(neg))
        raise ValueError("Harmonization failed due to negative non-CO2 gases")

    return report
