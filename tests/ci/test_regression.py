import pytest
import pandas as pd
import pandas.util.testing as pdt

# TODO: this should be parameterized by a common config file for all tests


def test_msg():
    exp_path = 'msg_harmonized.xlsx'
    obs_path = 'test-msg/msg_harmonized.xlsx'
    if not os.path.exists(exp_path):
        pytest.skip('Expected file does not exist: {}'.format(exp_path)

    obs=pd.read_excel(obs_path, sheet_name='data')
    exp=pd.read_excel(exp_path, sheet_name='data')

    pdt.assert_frame_equal(obs, exp)


def test_gcam():
    exp_path='gcam_harmonized.xlsx'
    obs_path='test-gcam/gcam_harmonized.xlsx'
    if not os.path.exists(exp_path):
        pytest.skip('Expected file does not exist: {}'.format(exp_path)

    obs=pd.read_excel(obs_path, sheet_name='data')
    exp=pd.read_excel(exp_path, sheet_name='data')

    pdt.assert_frame_equal(obs, exp)
