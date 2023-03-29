import os
import shutil
from os.path import join

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from aneris import cli


# This is a class that runs all tests through the harmonize CLI Note that it
# uses the actual harmonize API rather than subprocessing the CLI because
# coveralls does not pick up the lines covered in CLI calls when in a docker
# container.
#
# I don't know why. I spent 4+ hours digging. I am done. I hope I never have to
# worry about this again.

here = join(os.path.dirname(os.path.realpath(__file__)))
ci_path = join(here, "ci")

# check variables for if we are on CI (will then run regression tests)
ON_CI_REASON = "No access to regression test credentials"
try:
    os.environ["ANERIS_CI_USER"]
    ON_CI = True
except KeyError:
    ON_CI = False

FILE_SUFFIXES = [
    "global_only",
    "regions_sectors",
    "global_sectors",
    "mock_pipeline_prototype",
    "pipeline_progress",
    "full_ar6",
    "global_ar6",
]


class TestHarmonizeRegression:
    def _run(self, inf, checkf, hist, reg, rc, prefix, name):
        # path setup
        prefix = join(here, prefix)
        hist = join(prefix, hist)
        reg = join(prefix, reg)
        rc = join(prefix, rc)
        inf = join(prefix, inf)
        outf = join(prefix, f"{name}_harmonized.xlsx")
        outf_meta = join(prefix, f"{name}_metadata.xlsx")
        outf_diag = join(prefix, f"{name}_diagnostics.xlsx")
        clean = [outf, outf_meta, outf_diag]

        # make sure we're fresh
        for f in clean:
            if os.path.exists(f):
                os.remove(f)

        # run
        print(inf, hist, reg, rc, name)
        cli.harmonize(
            inf,
            hist,
            reg,
            rc,
            prefix,
            name,
            return_result=False,
        )

        # test
        ncols = 5
        expfile = join(prefix, checkf)
        exp = pd.read_excel(
            expfile,
            sheet_name="data",
            index_col=list(range(ncols)),
            engine="openpyxl",
        ).sort_index()
        exp.columns = exp.columns.astype(str)
        obs = pd.read_excel(
            outf,
            sheet_name="data",
            index_col=list(range(ncols)),
            engine="openpyxl",
        ).sort_index()
        assert_frame_equal(exp, obs, check_dtype=False)

        # tidy up after
        for f in clean:
            if os.path.exists(f):
                os.remove(f)

    @pytest.mark.parametrize("file_suffix", FILE_SUFFIXES)
    def test_basic_run(self, file_suffix):
        # this is run no matter what
        prefix = "test_data"
        checkf = f"test_{file_suffix}.xlsx"
        hist = f"history_{file_suffix}.xls"
        reg = f"regions_{file_suffix}.csv"
        inf = f"model_{file_suffix}.xls"
        rc = f"aneris_{file_suffix}.yaml"

        # get all arguments
        self._run(inf, checkf, hist, reg, rc, prefix, file_suffix)

    @pytest.mark.skipif(not ON_CI, reason=ON_CI_REASON)
    @pytest.mark.parametrize("name", ["msg", "gcam"])
    def test_regression_ci(self, name):
        prefix = join(ci_path, f"test-{name}")
        checkf = f"{name}_harmonized.xlsx"
        hist = "history.csv"
        reg = "regiondef.xlsx"
        rc = "rc.yaml"
        inf = "inputfile.xlsx"

        # copy needed files
        for fname in [hist, rc, checkf]:
            src = join(ci_path, fname)
            dst = join(prefix, fname)
            shutil.copyfile(src, dst)

        # get all arguments
        self._run(inf, checkf, hist, reg, rc, prefix, name)
