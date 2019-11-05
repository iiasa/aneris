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

    def _run(self, inf, checkf, hist, reg, rc, prefix, name):
        # path setup
        prefix = join(here, prefix)
        hist = join(prefix, hist)
        reg = join(prefix, reg)
        rc = join(prefix, rc)
        inf = join(prefix, inf)
        outf = join(prefix, '{}_harmonized.xlsx'.format(name))
        outf_meta = join(prefix, '{}_metadata.xlsx'.format(name))
        outf_diag = join(prefix, '{}_diagnostics.xlsx'.format(name))
        clean = [outf, outf_meta, outf_diag]

        # make sure we're fresh
        for f in clean:
            if os.path.exists(f):
                os.remove(f)

        # run
        print(inf, hist, reg, rc, name)
        cli.harmonize(inf, hist, reg, rc, prefix, name)

        # test
        xfile = join(prefix, checkf)
        x = pd.read_excel(xfile, sheet_name='data')
        y = pd.read_excel(outf, sheet_name='data')
        assert_frame_equal(x, y)

        # tidy up after
        for f in clean:
            if os.path.exists(f):
                os.remove(f)

    @pytest.mark.parametrize("file_suffix", ['global_only', 'global_sectors', 'regions_sectors'])
    def test_basic_run(self, file_suffix):
        # this is run no matter what
        prefix = 'test_data'
        checkf = 'test_{}.xlsx'.format(file_suffix)
        hist = 'history_{}.xls'.format(file_suffix)
        reg = 'regions_{}.csv'.format(file_suffix)
        inf = 'model_{}.xls'.format(file_suffix)
        rc = 'aneris_{}.yaml'.format(file_suffix)

        # get all arguments
        self._run(inf, checkf, hist, reg, rc, prefix, file_suffix)

    @pytest.mark.skipif(not ON_CI, reason=ON_CI_REASON)
    @pytest.mark.parametrize("name", ['msg', 'gcam'])
    def test_regression_ci(self, name):
        prefix = join(ci_path, 'test-{}'.format(name))
        checkf = '{}_harmonized.xlsx'.format(name)
        hist = 'history.csv'
        reg = 'regiondef.xlsx'
        rc = 'rc.yaml'
        inf = 'inputfile.xlsx'

        # copy needed files
        for fname in [hist, rc, checkf]:
            src = join(ci_path, fname)
            dst = join(prefix, fname)
            shutil.copyfile(src, dst)

        # get all arguments
        self._run(inf, checkf, hist, reg, rc, prefix, name)
