import collections
import os
import yaml

from .utils import isstr

_rc_defaults = """
config:
    default_luc_method: reduce_ratio_2150_cov
    cov_threshold: 20
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

    def __init__(self, io=None):
        self.store = dict()
        io = io or {}
        if isstr(io):
            if os.path.exists(io):
                with open(io) as f:
                    io = yaml.safe_load(f)
            else:
                io = yaml.safe_load(io)

        defaults = yaml.safe_load(_rc_defaults)
        opts = _recursive_update(defaults, io)
        self.store.update(opts)

    def __getitem__(self, k):
        return self.store[k]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
