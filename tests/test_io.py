
from aneris import io

_defaults = {
    'config': {
        'default_luc_method': 'reduce_ratio_2150_cov',
        'cov_threshold': 20,
    },
}


def test_default_rc():
    exp = _defaults
    obs = io.RunControl()
    for k in exp.keys():
        assert k in obs
        assert exp[k] == obs[k]


def test_nondefault_rc():
    rcstr = """
    config:
        cov_threshold: 42
    """

    obs = io.RunControl(rcstr)
    exp = _defaults
    exp['config']['cov_threshold'] = 42
    for k in exp.keys():
        assert k in obs
        assert exp[k] == obs[k]
