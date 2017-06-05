"""Provides helper functions for reading input data and configuration files.

The default configuration values are provided in aneris.RC_DEFAULTS.
"""
import collections
import os
import yaml

import pandas as pd

from aneris.utils import isstr, isnum, iamc_idx

RC_DEFAULTS = """
config:
    default_luc_method: reduce_ratio_2150_cov
    cov_threshold: 20
    harmonize_year: 2015
prefix: CEDS+|9+ Sectors
suffix: Unharmonized
add_5regions: true
"""


def _read_data(indfs):
    datakeys = sorted([x for x in indfs if x.startswith('data')])
    df = pd.concat([indfs[k] for k in datakeys])
    # don't know why reading from excel changes dtype and column types
    # but I have to reset them manually
    df.columns = df.columns.astype(str)
    numcols = [x for x in df.columns if isnum(x)]
    df[numcols] = df[numcols].astype(float)

    # some teams also don't provide standardized column names and styles
    df.columns = df.columns.str.capitalize()

    return df


def _recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = _recursive_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


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
    if f.endswith('csv'):
        df = pd.read_csv(f, *args, **kwargs)
    else:
        df = pd.read_excel(f, *args, **kwargs)

    if str_cols:
        df.columns = [str(x) for x in df.columns]

    return df


def pd_write(df, f, *args, **kwargs):
    """Try to write a file with pandas, supports CSV and XLSX"""
    # guess whether to use index, unless we're told otherwise
    index = kwargs.pop('index', isinstance(df.index, pd.MultiIndex))

    if f.endswith('csv'):
        df.to_csv(f, index=index, *args, **kwargs)
    else:
        writer = pd.ExcelWriter(f, engine='xlsxwriter')
        df.to_excel(writer, index=index, *args, **kwargs)
        writer.save()


def read_excel(f):
    """Read an excel-based input file for harmonization.

    Parameters
    ----------
    f : string
        path to input file

    Returns
    -------
    model : pd.DataFrame
        model data frame in IAMC format
    overrides : pd.DataFrame
        overrides data frame in IAMC format
    config : dictionary
        configuration overrides (if any)
    """
    indfs = pd_read(f, sheetname=None, encoding='utf-8')
    model = _read_data(indfs)

    # make an empty df which will be caught later
    overrides = indfs['harmonization'] if 'harmonization' in indfs \
        else pd.DataFrame([], columns=iamc_idx + ['Unit'])

    # get run control
    config = {}
    if'Configuration' in overrides:
        config = overrides[['Configuration', 'Value']].dropna()
        config = config.set_index('Configuration').to_dict()['Value']
        overrides = overrides.drop(['Configuration', 'Value'], axis=1)

    # a single row of nans implies only configs provided,
    # if so, only return the empty df
    if len(overrides) == 1 and overrides.isnull().values.all():
        overrides = pd.DataFrame([], columns=iamc_idx + ['Unit'])

    return model, overrides, config


class RunControl(collections.Mapping):
    """A thin wrapper around a Python Dictionary to support configuration of
    harmonization execution. Input can be provided as dictionaries or YAML
    files.
    """

    def __init__(self, rc=None, defaults=None):
        """
        Parameters
        ----------
        rc : string, file, dictionary, optional
            a path to a YAML file, a file handle for a YAML file, or a 
            dictionary describing run control configuration
        defaults : string, file, dictionary, optional
            a path to a YAML file, a file handle for a YAML file, or a 
            dictionary describing **default** run control configuration
        """
        rc = rc or {}
        defaults = defaults or RC_DEFAULTS

        rc = self._load_yaml(rc)
        defaults = self._load_yaml(defaults)
        self.store = _recursive_update(defaults, rc)

    def __getitem__(self, k):
        return self.store[k]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return self.store.__repr__()

    def _get_path(self, key, fyaml, fname):
        if os.path.exists(fname):
            return fname

        _fname = os.path.join(os.path.dirname(fyaml), fname)
        if not os.path.exists(_fname):
            msg = "YAML key '{}' in {}: {} is not a valid relative " + \
                "or absolute path"
            raise IOError(msg.format(key, fyaml, fname))
        return _fname

    def _fill_relative_paths(self, fyaml, d):
        file_keys = [
            'exogenous',
        ]
        for k in file_keys:
            if k in d:
                d[k] = [self._get_path(k, fyaml, fname) for fname in d[k]]

    def _load_yaml(self, obj):
        check_rel_paths = False
        if hasattr(obj, 'read'):  # it's a file
            obj = obj.read()
        if isstr(obj) and os.path.exists(obj):
            check_rel_paths = True
            fname = obj
            with open(fname) as f:
                obj = f.read()
        if not isinstance(obj, dict):
            obj = yaml.load(obj)
        if check_rel_paths:
            self._fill_relative_paths(fname, obj)
        return obj

    def recursive_update(self, k, d):
        """Recursively update a top-level option in the run control

        Parameters
        ----------
        k : string
            the top-level key
        d : dictionary or similar
            the dictionary to use for updating
        """
        u = self.__getitem__(k)
        self.store[k] = _recursive_update(u, d)
