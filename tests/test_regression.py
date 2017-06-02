import pytest
import subprocess
import os

import pandas as pd
from pandas.util.testing import assert_frame_equal


here = os.path.join(os.path.dirname(os.path.realpath(__file__)))
cmd = 'aneris {} --history {} --regions {} --rc {} --output_path {} --output_prefix test'


slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)


class TestRegression():

    def _run(self, prefix, inf, checkf, reg='message.csv'):
        prefix = os.path.join(here, prefix)
        add_prefix = lambda f: os.path.join(prefix, f)
        hist = add_prefix('history.csv')
        reg = add_prefix(reg)
        rc = add_prefix('aneris.yaml')

        inf = add_prefix(inf)
        outf = add_prefix('test_harmonized.xlsx')
        if os.path.exists(outf):
            os.remove(outf)

        _cmd = cmd.format(inf, hist, reg, rc, prefix)
        print(_cmd)
        subprocess.check_call(_cmd.split())

        xfile = os.path.join(prefix, checkf)
        x = pd.read_excel(xfile, sheetname='data')
        y = pd.read_excel(outf, sheetname='data')

        assert_frame_equal(x, y)

    def test_basic_run(self):
        prefix = 'test_data'
        inf = 'model.xlsx'
        checkf = 'test_basic_run.xlsx'
        self._run(prefix, inf, checkf, reg='regions.csv')

    @slow
    def test_message_ref(self):
        prefix = 'regression_data'
        inf = 'MESSAGE-GLOBIOM_SSP2-Ref-SPA0-V25_unharmonized.xlsx'
        checkf = 'test_regress_ssp2_ref.xlsx'
        self._run(prefix, inf, checkf)

    @slow
    def test_message_no_sheet(self):
        prefix = 'regression_data'
        inf = 'no_sheet.xlsx'
        checkf = 'test_regress_ssp2_no_sheet.xlsx'
        self._run(prefix, inf, checkf)

    @slow
    def test_message_empty_sheet(self):
        prefix = 'regression_data'
        inf = 'empty_sheet.xlsx'
        checkf = 'test_regress_ssp2_empty_sheet.xlsx'
        self._run(prefix, inf, checkf)
