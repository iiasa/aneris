import pytest
import os

import pandas as pd
from pandas.util.testing import assert_frame_equal

from aneris import cli

# decorator for slow-running tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)


# This is a class that runs all tests through the harmonize CLI Note that it
# uses the actual harmonize API rather than subprocessing the CLI because
# coveralls does not pick up the lines covered in CLI calls when in a docker
# container.
#
# I don't know why. I spent 4+ hours digging. I am done. I hope I never have to
# worry about this again.
class TestHarmonizeRegression():

    def _run(self, prefix, inf, checkf, hist='history.csv', reg='message.csv'):
        # path setup
        here = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        prefix = os.path.join(here, prefix)
        add_prefix = lambda f: os.path.join(prefix, f)

        # get all arguments
        hist = add_prefix(hist)
        reg = add_prefix(reg)
        rc = add_prefix('aneris.yaml')
        inf = add_prefix(inf)
        outf = add_prefix('test_harmonized.xlsx')
        if os.path.exists(outf):
            os.remove(outf)

        # run
        print(inf, hist, reg, rc, prefix, 'test')
        cli.harmonize(inf, hist, reg, rc, prefix, 'test')

        # test
        xfile = os.path.join(prefix, checkf)
        x = pd.read_excel(xfile, sheetname='data')
        y = pd.read_excel(outf, sheetname='data')
        assert_frame_equal(x, y)

    def test_basic_run(self):
        prefix = 'test_data'
        inf = 'model.xls'
        checkf = 'test_basic_run.xlsx'
        self._run(prefix, inf, checkf, hist='history.xls', reg='regions.csv')

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
