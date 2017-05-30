import collections
import os
import yaml

from aneris.utils import isstr

_rc_defaults = """
config:
    default_luc_method: reduce_ratio_2150_cov
    cov_threshold: 20
    harmonize_year: 2015
prefix: CEDS+|9+
suffix: Unharmonized
"""


def _recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = _recursive_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


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

    def __getitem__(self, k):
        return self.store[k]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return self.store.__repr__()
