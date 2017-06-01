import subprocess
import os

import pandas as pd
from pandas.util.testing import assert_frame_equal

here = os.path.join(os.path.dirname(os.path.realpath(__file__)))


prefix = os.path.join(here, 'regression_data')
add_prefix = lambda f: os.path.join(prefix, f)
hist = add_prefix('history.csv')
reg = add_prefix('message.csv')
rc = add_prefix('aneris.yaml')
cmd = 'aneris {} --history {} --regions {} --rc {} --output_path {} --output_prefix test'


class TestRegression():

    def _run(self, inf, checkf):
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

    def test_message_ref(self):
        inf = 'MESSAGE-GLOBIOM_SSP2-Ref-SPA0-V25_unharmonized.xlsx'
        checkf = 'test_regress_ssp2_ref.xlsx'
        self._run(inf, checkf)

    # def test_message_no_sheet(self):
    #     inf = 'no_sheet.xlsx'
    #     checkf = 'test_regress_ssp2_no_sheet.xlsx'
    #     self._run(inf, checkf)

    # def test_message_empty_sheet(self):
    #     inf = 'empty_sheet.xlsx'
    #     checkf = 'test_regress_ssp2_empty_sheet.xlsx'
    #     self._run(inf, checkf)
