import pytest
import os
import shutil

import pandas as pd

from pandas.util.testing import assert_frame_equal
from os.path import join

from aneris import cli

# This is a class that runs all tests through the harmonize CLI Note that it
# uses the actual harmonize API rather than subprocessing the CLI because
# coveralls does not pick up the lines covered in CLI calls when in a docker
# container.
#
# I don't know why. I spent 4+ hours digging. I am done. I hope I never have to
# worry about this again.

here = join(os.path.dirname(os.path.realpath(__file__)))
ci_path = join(here, 'ci')

# check variables for if we are on CI (will then run regression tests)
ON_CI_REASON = 'No access to regression test credentials'
try:
    os.environ['ANERIS_CI_USER']
    ON_CI = True
except KeyError:
    ON_CI = False


class TestHarmonizeRegression():

    def _run(self, inf, checkf, hist, reg, rc, outf, prefix):
        # path setup
        prefix = join(here, prefix)
        hist = join(prefix, hist)
        reg = join(prefix, reg)
        rc = join(prefix, rc)
        inf = join(prefix, inf)
        outf = join(prefix, outf)

        if os.path.exists(outf):
            os.remove(outf)

        # run
        print(inf, hist, reg, rc, 'test')
        cli.harmonize(inf, hist, reg, rc, prefix, 'test')

        # test
        xfile = join(prefix, checkf)
        x = pd.read_excel(xfile, sheet_name='data')
        y = pd.read_excel(outf, sheet_name='data')
        assert_frame_equal(x, y)

        clean = [
            outf,
            join(prefix, 'test_metadata.xlsx'),
            join(prefix, 'test_diagnostics.xlsx'),
        ]
        for f in clean:
            if os.path.exists(f):
                os.remove(f)

    def test_basic_run(self):
        # this is run no matter what
        prefix = 'test_data'
        checkf = 'test_basic_run.xlsx'
        hist = 'history.xls'
        reg = 'regions.csv'
        rc = 'aneris.yaml'
        inf = 'model.xls'
        outf = 'test_harmonized.xlsx'

        # get all arguments
        self._run(inf, checkf, hist, reg, rc, outf, prefix)

    #
    # the following are run only on CI, this should be parameterized in the
    # future
    #

    def _run_ci(self, name):
        prefix = join(ci_path, 'test-{}'.format(name))
        checkf = '{}_harmonized.xlsx'.format(name)
        hist = 'history.csv'
        reg = 'regiondef.xlsx'
        rc = 'rc.yaml'
        inf = 'inputfile.xlsx'
        outf = 'test_harmonized.xlsx'

        # copy needed files
        for fname in [hist, rc, checkf]:
            src = join(ci_path, fname)
            dst = join(prefix, fname)
            shutil.copyfile(src, dst)

        # get all arguments
        self._run(inf, checkf, hist, reg, rc, outf, prefix)

    # only runs if access to regression data is available
    @pytest.mark.skipif(not ON_CI, reason=ON_CI_REASON)
    def test_msg(self):
        # file setup
        name = 'msg'
        self._run_ci(name)

    # only runs if access to regression data is available
    @pytest.mark.skipif(not ON_CI, reason=ON_CI_REASON)
    def test_gcam(self):
        # file setup
        name = 'gcam'
        self._run_ci(name)
