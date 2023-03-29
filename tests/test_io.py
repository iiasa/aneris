import tempfile

import pytest

from aneris import _io


_defaults = {
    "config": {
        "default_luc_method": "reduce_ratio_2150_cov",
        "default_offset_method": "reduce_offset_2080",
        "default_ratio_method": "reduce_ratio_2080",
        "cov_threshold": 20,
        "harmonize_year": 2015,
        "global_harmonization_only": False,
        "replace_suffix": "Harmonized-DB",
    },
    "prefix": "CEDS+|9+ Sectors",
    "suffix": "Unharmonized",
    "add_5regions": True,
}


def test_default_rc():
    exp = _defaults
    obs = _io.RunControl()
    assert obs == exp


def test_mutable():
    obs = _io.RunControl()
    with pytest.raises(TypeError):
        obs["foo"] = "bar"


def test_nondefault_rc():
    rcstr = """
    config:
        cov_threshold: 42
    """

    obs = _io.RunControl(rcstr)
    exp = _defaults
    exp["config"]["cov_threshold"] = 42
    assert exp == obs


def test_nondefault_rc_file_read():
    rcstr = b"""
    config:
        cov_threshold: 42
    """
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(rcstr)
        f.flush()
        obs = _io.RunControl(f.name)
        exp = _defaults
        exp["config"]["cov_threshold"] = 42
        assert exp == obs


def test_recursive_update():
    update = {
        "foo": "bar",
        "cov_threshold": 42,
    }
    exp = _defaults
    exp["config"].update(update)

    obs = _io.RunControl()
    obs.recursive_update("config", update)
    assert obs == exp
