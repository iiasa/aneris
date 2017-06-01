import subprocess
import os

import pandas as pd
from pandas.util.testing import assert_frame_equal

here = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def test_message_ref():
    prefix = os.path.join(here, 'regression_data')
    add_prefix = lambda f: os.path.join(prefix, f)
    _inf = 'MESSAGE-GLOBIOM_SSP2-Ref-SPA0-V25_unharmonized.xlsx'
    outf = add_prefix('test_harmonized.xlsx')
    if os.path.exists(outf):
        os.remove(outf)

    inf = add_prefix(_inf)
    hist = add_prefix('history.csv')
    reg = add_prefix('message.csv')
    rc = add_prefix('aneris.yaml')
    cmd = 'aneris {} --history {} --regions {} --rc {} --output_path {} --output_prefix test'.format(
        inf, hist, reg, rc, prefix)
    print(cmd)
    subprocess.check_call(cmd.split())

    xfile = os.path.join(prefix, 'test_regress_ssp2_ref.xlsx')
    x = pd.read_excel(xfile, sheetname='data')
    y = pd.read_excel(outf, sheetname='data')

    assert_frame_equal(x, y)
