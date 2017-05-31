import subprocess
import os

import pandas as pd
from pandas.util.testing import assert_frame_equal

here = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def test_message_ref():
    prefix = os.path.join(here, '..', 'tmp')
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

    xfile = _inf.replace('unharmonized', 'unharmonized_harmonized')
    xpth = os.path.join(here, '..', '..', 'harmonization')
    x = pd.read_excel(os.path.join(xpth, xfile), sheetname='data')
    y = pd.read_excel(outf, sheetname='data')

    assert_frame_equal(x, y)
