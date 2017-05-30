import pytest
import tempfile

from aneris import _io

_defaults = {
    'config': {
        'default_luc_method': 'reduce_ratio_2150_cov',
        'cov_threshold': 20,
        'harmonize_year': 2015,
    },
}


def test_default_rc():
    exp = _defaults
    obs = _io.RunControl()
    assert obs == exp


def test_mutable():
    obs = _io.RunControl()
    with pytest.raises(TypeError):
        obs['foo'] = 'bar'


def test_nondefault_rc():
    rcstr = """
    config:
        cov_threshold: 42
    """

    obs = _io.RunControl(rcstr)
    exp = _defaults
    exp['config']['cov_threshold'] = 42
    assert exp == obs


def test_nondefault_rc_file_read():
    rcstr = """
    config:
        cov_threshold: 42
    """
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(rcstr)
        f.flush()
        obs = _io.RunControl(f.name)
        exp = _defaults
        exp['config']['cov_threshold'] = 42
        assert exp == obs
