import collections
import os
import yaml

from aneris.utils import isstr

_rc_defaults = """
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

    if '2015' not in df.columns:
        msg = 'Base year not found in model data. Existing columns are {}.'
        raise ValueError(msg.format(df.columns))

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


def pd_read(f, *args, **kwargs):
    """Try to read a file with pandas, no fancy stuff"""
    if f.endswith('csv'):
        return pd.read_csv(f, *args, **kwargs)
    else:
        return pd.read_excel(f, *args, **kwargs)


def pd_write(df, f, *args, **kwargs):
    # guess whether to use index, unless we're told otherwise
    index = kwargs.pop('index', isinstance(df.index, pd.MultiIndex))

    if f.endswith('csv'):
        df.to_csv(f, index=index, *args, **kwargs)
    else:
        writer = pd.ExcelWriter(f, engine='xlsxwriter')
        df.to_excel(writer, index=index, *args, **kwargs)
        writer.save()


def read_excel(f):
    indfs = utils.pd_read(f, sheetname=None, encoding='utf-8')
    model = _read_data(indfs)

    # make an empty df which will be caught later
    overrides = indfs['harmonization'] if 'harmonization' in indfs \
        else pd.DataFrame([], columns=['Scenario'])

    # get run control
    config = {}
    if'Configuration' in overrides:
        config = overrides[['Configuration', 'Value']].dropna()
        config = config.set_index('Configuration').to_dict()['Value']
        overrides = overrides.drop(['Configuration', 'Value'], axis=1)

    return model, overrides, config


class RunControl(collections.Mapping):

    def __init__(self, rc=None):
        self.store = dict()
        rc = rc or {}
        if isinstance(rc, file):
            rc = rc.read()
        if isstr(rc) and os.path.exists(rc):
            with open(rc) as f:
                rc = f.read()
        if not isinstance(rc, dict):
            rc = yaml.load(rc)
        defaults = yaml.safe_load(_rc_defaults)
        opts = _recursive_update(defaults, rc)
        self.store.update(opts)

    def recursive_update(self, k, d):
        u = self.__getitem__(k)
        self.store[k] = _recursive_update(u, d)

    def __getitem__(self, k):
        return self.store[k]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return self.store.__repr__()
