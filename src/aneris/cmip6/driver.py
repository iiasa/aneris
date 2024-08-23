import numpy as np
import pandas as pd
from pandas_indexing import assignlevel, isin

import aneris.cmip6.cmip6_utils as cmip6_utils
import aneris.utils as utils
from aneris.harmonize import Harmonizer, _log, _warn
from aneris.utils import pd_read


class _TrajectoryPreprocessor:
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
        _log(f"Downselecting {select} variables")

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
        xlator = cmip6_utils.FormatTranslator()

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
            idx = list(set(utils.df_idx) - set(["unit"]))
            self.overrides = (
                xlator.to_std(df=self.overrides.copy(), set_metadata=False, unit=False)
                .set_index(idx)
                .sort_index()
            )
            self.overrides.columns = self.overrides.columns.str.lower()
            self.overrides = self.overrides["method"]

    def _agg_hist(self):
        # aggregate and clean hist
        _log("Aggregating historical values to native regions")
        # must set verify to false for now because some isos aren't included!
        self.hist = cmip6_utils.agg_regions(
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


class HarmonizationDriver:
    """
    A helper class to harmonize all scenarios for a model.
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
            _log(f"Manually adding global regional definition: {glb}")
            self.regions = self.regions.append(glb, ignore_index=True)

        model_names = self.model.Model.unique()
        if len(model_names) > 1:
            raise ValueError("Can not have more than one model to harmonize")
        self.model_name = model_names[0]
        self._xlator = cmip6_utils.FormatTranslator(
            prefix=self.prefix, suffix=self.suffix
        )
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
        """
        Harmonize a given scneario. Get results from
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
        self._meta = assignlevel(self._meta, model=self.model_name, scenario=scenario)
        self._postprocess_trajectories(scenario)

        # store results
        self._model_dfs.append(self._model)
        self._metadata_dfs.append(self._meta)
        self._diagnostic_dfs.append(self._diag)

    def scenarios(self):
        """
        Return all known scenarios.
        """
        return self.model["Scenario"].unique()

    def harmonized_results(self):
        """
        Return 3-tuple of (pd.DataFrame of harmonized trajectories,
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
    gases = cmip6_utils.harmonize_total_gases if default_global_gases else all_gases
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

    cmip6_utils.check_null(m, "model")
    cmip6_utils.check_null(h, "hist", fail=True)
    harmonizer = Harmonizer(m, h, config=config)
    _log("Harmonizing (with example methods):")
    _log(harmonizer.methods(year=harmonizer.base_year, overrides=o).head())
    if o is not None:
        _log("and override methods:")
        _log(o.head())
    m = harmonizer.harmonize(year=harmonizer.base_year, overrides=o)
    cmip6_utils.check_null(m, "model")

    metadata = harmonizer.metadata()
    return m, metadata


def _harmonize_regions(
    config, prefix, suffix, regions, hist, model, overrides, base_year, add_5regions
):
    # clean model
    model = cmip6_utils.subtract_regions_from_world(model, "model", base_year)
    model = cmip6_utils.remove_recalculated_sectors(model, prefix, suffix)
    # remove rows with all 0s
    model = model[(model.T > 0).any()]

    # clean hist
    hist = cmip6_utils.subtract_regions_from_world(hist, "hist", base_year)
    hist = cmip6_utils.remove_recalculated_sectors(hist, prefix, suffix)

    # remove rows with all 0s
    hist = hist[(hist.T > 0).any()]

    if model.empty:
        raise RuntimeError("Model is empty after downselecting regional values")

    # harmonize
    cmip6_utils.check_null(model, "model")
    cmip6_utils.check_null(hist, "hist", fail=True)
    harmonizer = Harmonizer(model, hist, config=config)
    _log("Harmonizing (with example methods):")
    _log(harmonizer.methods(overrides=overrides).head())

    if overrides is not None:
        _log("and override methods:")
        _log(overrides.head())
    model = harmonizer.harmonize(overrides=overrides)
    cmip6_utils.check_null(model, "model")
    metadata = harmonizer.metadata()

    # add aggregate variables. this works in three steps:
    # step 1: remove any sector total trajectories that also have subsectors to
    # be recalculated
    idx = cmip6_utils.recalculated_row_idx(model, prefix, suffix)
    if idx.any():
        msg = "Removing sector aggregates. Recalculating with harmonized totals."
        _warn(msg)
        model = model[~idx]
    totals = "|".join([prefix, suffix])
    sector_total_idx = isin(model, sector=totals)
    subsector_idx = ~sector_total_idx
    # step 2: on the "clean" df, recalculate those totals
    subsectors_with_total_df = (
        cmip6_utils.EmissionsAggregator(model[subsector_idx])
        .add_variables(totals=totals, aggregates=False)
        .df.set_index(utils.df_idx)
    )
    # step 3: recombine with model data that was sector total only
    sector_total_df = model[sector_total_idx]
    model = pd.concat([sector_total_df, subsectors_with_total_df])
    cmip6_utils.check_null(model, "model")

    # combine regional values to send back into template form
    model.reset_index(inplace=True)
    model = model.set_index(utils.df_idx).sort_index()
    glb = cmip6_utils.combine_rows(
        model, "region", "World", sumall=False, rowsonly=True
    )
    model = glb.combine_first(model)

    # add 5regions
    if add_5regions:
        _log("Adding 5region values")
        # explicitly don't add World, it already exists from aggregation
        mapping = regions[regions["Native Region Code"] != "World"].copy()
        aggdf = cmip6_utils.agg_regions(
            model, mapping=mapping, rfrom="Native Region Code", rto="5_region"
        )
        model = pd.concat([model, aggdf])
        assert not model.isnull().any(axis=None)

    # duplicates come in from World and World being translated
    duplicates = model.index.duplicated(keep="first")
    if duplicates.any():
        regions = model[duplicates].index.get_level_values("region").unique()
        msg = f"Dropping duplicate rows found for regions: {regions}"
        _warn(msg)
        model = model[~duplicates]

    return model, metadata


def diagnostics(unharmonized, model, metadata, config=None):
    """
    Provide warnings or throw errors based on harmonized model data and
    metadata.

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
        _warn(f"LARGE MISSING Values Found!!:\n {report}")

    #
    # report on large medium an dlong-term differences
    #
    cols = utils.numcols(model)
    report = model.copy()
    mid, end = cols[len(cols) // 2 - 1], cols[-1]

    if "mid" in config:
        bigmid = np.abs(model[mid] - unharmonized[mid]) / unharmonized[mid]
        bigmid = bigmid[bigmid > config["mid"]]
        report[f"{mid}_diff"] = bigmid

    if "end" in config:
        bigend = np.abs(model[end] - unharmonized[end]) / unharmonized[end]
        bigend = bigend[bigend > config["end"]]
        report[f"{end}_diff"] = bigend

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
        _warn(f"Negative Emissions found for non-CO2 gases:\n {neg}")
        raise ValueError("Harmonization failed due to negative non-CO2 gases")

    return report
