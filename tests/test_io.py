import pytest
import tempfile

from aneris import io

_defaults = {
    'config': {
        'default_luc_method': 'reduce_ratio_2150_cov',
        'cov_threshold': 20,
        'harmonize_year': 2015,
    },
}


def test_default_rc():
    exp = _defaults
    obs = io.RunControl()
    assert exp == obs


def test_mutable():
    obs = io.RunControl()
    with pytest.raises(TypeError):
        obs['foo'] = 'bar'


def test_nondefault_rc():
    rcstr = """
    config:
        cov_threshold: 42
    """

    obs = io.RunControl(rcstr)
    exp = _defaults
    exp['config']['cov_threshold'] = 42
    assert exp == obs


def test_nondefault_rc_file():
    rcstr = """
    config:
        cov_threshold: 42
    """
    with tempfile.TemporaryFile() as f:
        print(f)
        f.write(rcstr)
        obs = io.RunControl(f)
        exp = _defaults
        exp['config']['cov_threshold'] = 42
        assert exp == obs
