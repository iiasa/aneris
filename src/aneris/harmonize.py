from functools import partial
from itertools import chain

import pandas as pd
from pandas_indexing import projectlevel, semijoin, uniquelevel

from aneris import utils
from aneris.errors import (
    MissingHarmonisationYear,
    MissingHistoricalError,
    MissingScenarioError,
)
from aneris.methods import (
    budget,
    coeff_of_var,
    constant_offset,
    constant_ratio,
    default_methods,
    harmonize_factors,
    hist_zero,
    linear_interpolate,
    model_zero,
    reduce_offset,
    reduce_ratio,
)


def _log(msg, *args, **kwargs):
    utils.logger().info(msg, *args, **kwargs)


def _warn(msg, *args, **kwargs):
    utils.logger().warning(msg, *args, **kwargs)


def _check_data(hist, scen, year, idx):
    # always check that unit exists
    if "unit" not in idx:
        idx += ["unit"]

    s = uniquelevel(scen, idx)
    h = uniquelevel(hist, idx)
    if h.empty:
        raise MissingHarmonisationYear("No historical data in harmonization year")

    if not s.difference(h).empty:
        raise MissingHistoricalError(
            "Historical data does not match scenario data in harmonization "
            f"year for\n {s.difference(h).to_frame().to_string(index=False, max_rows=100)}"
        )

    if not h.difference(s).empty:
        raise MissingScenarioError(
            "Scenario data does not match historical data in harmonization "
            f"year for\n {h.difference(s).to_frame().to_string(index=False, max_rows=100)}"
        )


def _check_overrides(overrides, data_index):
    if overrides is None:
        return

    if not isinstance(overrides, pd.Series):
        raise TypeError("Overrides required to be pd.Series")

    if not overrides.name == "method":
        raise ValueError("Overrides name must be method")

    # Check whether there exists an override for at least one data variable
    _, lidx, _ = overrides.index.join(data_index, how="right", return_indexers=True)
    if lidx is None:
        return

    if (lidx == -1).all():
        raise ValueError(
            "overrides must have at least one index dimension "
            f"aligned with methods: {data_index.names}"
        )


class Harmonizer:
    """
    A class used to harmonize model data to historical data in the standard
    calculation format.
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
        self,
        data,
        history,
        config={},
        harm_idx=["region", "gas", "sector"],
        method_choice=None,
    ):
        """
        The Harmonizer class prepares and harmonizes historical data to model
        data.

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
                "Data to harmonize exceeds historical data avaiablility:\n"
                f"{data_check.difference(hist_check)}"
            )

        def check_idx(df, label):
            final_idx = harm_idx + ["unit"]
            extra_idx = list(set(df.index.names) - set(final_idx))
            if extra_idx:
                df = df.droplevel(extra_idx)
                _warn(f"Extra index found in {label}, dropping levels {extra_idx}")
            return df

        data = check_idx(data, "data")
        history = check_idx(history, "history")
        history.columns = history.columns.astype(data.columns.dtype)

        # set basic attributes
        self.data = data[utils.numcols(data)]
        self.history = history
        self.methods_used = None

        # set up defaults
        self.base_year = (
            str(config["harmonize_year"]) if "harmonize_year" in config else None
        )
        self.method_choice = method_choice

        # set default methods to use in decision tree
        self.ratio_method = config.get("default_ratio_method")
        self.offset_method = config.get("default_offset_method")
        self.luc_method = config.get("default_luc_method")
        self.luc_cov_threshold = config.get("luc_cov_threshold")

    def metadata(self, year=None):
        """
        Return pd.DataFrame of method choice metadata.
        """
        base_year = year if year is not None else self.base_year or 2015

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
                self.history[base_year],
                self.history.apply(coeff_of_var, axis=1),
                self.data[base_year],
                self.model[base_year],
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

        kwargs = {
            "method_choice": self.method_choice,
            "ratio_method": self.ratio_method,
            "offset_method": self.offset_method,
            "luc_method": self.luc_method,
            "luc_cov_threshold": self.luc_cov_threshold,
        }
        methods, diagnostics = default_methods(
            self.history.droplevel(
                list(set(self.history.index.names) - set(self.harm_idx))
            ),
            self.data.droplevel(list(set(self.data.index.names) - set(self.harm_idx))),
            year,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
        return methods

    def _harmonize(self, method, idx, check_len, base_year):
        # get data
        model = semijoin(self.data, idx, how="right")
        hist = semijoin(self.history, idx, how="right")
        offsets = semijoin(self.offsets, idx, how="right")
        ratios = semijoin(self.ratios, idx, how="right")

        # get delta
        delta = hist if method == "budget" else ratios if "ratio" in method else offsets

        # checks
        assert not model.isnull().any(axis=None)
        assert not hist.isnull().any(axis=None)
        assert not delta.isnull().any(axis=None)
        if check_len:
            assert (len(model) < len(self.data)) & (len(hist) < len(self.history))

        # harmonize
        model = Harmonizer._methods[method](model, delta, harmonize_year=base_year)

        if model.isnull().any(axis=None):
            msg = "{} method produced NaNs: {}, {}"
            where = model.isnull().any(axis=1)
            raise ValueError(
                msg.format(method, model.loc[where, base_year], delta.loc[where])
            )

        # construct the full df of history and future
        return model

    def methods(self, year=None, overrides=None):
        # TODO: next issue is that other 'convenience' methods have less
        # robust override indices. need to decide how to support this
        """
        Return pd.DataFrame of methods to use for harmonization given
        pd.DataFrame of overrides.
        """
        # get method listing
        base_year = year if year is not None else self.base_year or 2015
        _check_overrides(overrides, self.data.index)
        methods = self._default_methods(year=base_year)

        if overrides is not None:
            # expand overrides index to match methods and align indicies
            overrides = semijoin(overrides, methods.index, how="right").reorder_levels(
                methods.index.names
            )
            if not overrides.index.difference(methods.index).empty:
                raise ValueError(
                    "Data to override exceeds model data avaiablility:\n"
                    f"{overrides.index.difference(methods.index)}"
                )
            overrides.name = methods.name

            # overwrite defaults with overrides
            methods = (
                overrides.combine_first(methods)
                .to_frame()
                .assign(default=methods, override=overrides)
            )

        return methods

    def harmonize(self, year=None, overrides=None):
        """
        Return pd.DataFrame of harmonized trajectories given pd.DataFrame
        overrides.
        """
        base_year = year if year is not None else self.base_year or 2015
        _check_data(self.history, self.data, base_year, self.harm_idx)

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
            self.model = pd.Series(
                index=self.data.index, name=base_year, dtype=float
            ).to_frame()
            msg = """Values found where model has positive and negative values
            and is zero in base year. Unsure how to proceed:\n{}\n{}"""
            cols = ["history", "unharmonized"]
            df1 = self.metadata(year=base_year).loc[methods == "unicorn", cols]
            df2 = self.data.loc[methods == "unicorn"]
            raise ValueError(msg.format(df1.reset_index(), df2.reset_index()))

        dfs = []
        y = base_year
        check_len = len(methods.unique()) > 1
        for method in methods.unique():
            _log(f"Harmonizing with {method}")
            # get subset indicies
            idx = methods[methods == method].index
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
