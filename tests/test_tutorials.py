import io
import os
import subprocess
import sys
import tempfile
import nbformat
import pytest
import jupyter

here = os.path.dirname(os.path.realpath(__file__))
tut_path = os.path.join(here, '..', 'doc', 'source')

# taken from the execellent example here:
# https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/


def _notebook_run(path, kernel=None, capsys=None):
    """Execute a notebook via nbconvert and collect output.
    :returns (parsed nb object, execution errors)
    """
    print(path)
    assert os.path.exists(path)
    major_version = sys.version_info[0]
    kernel = kernel or 'python{}'.format(major_version)
    if capsys is not None:
        with capsys.disabled():
            print('using py version {} with kernerl {}'.format(
                major_version, kernel))
    dirname, __ = os.path.split(path)
    os.chdir(dirname)
    fname = os.path.join(here, 'test.ipynb')
    args = [
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--ExecutePreprocessor.timeout=60',
        '--ExecutePreprocessor.kernel_name={}'.format(kernel),
        "--output", fname, path]
    subprocess.check_call(args)

    nb = nbformat.read(io.open(fname, encoding='utf-8'),
                       nbformat.current_nbformat)

    errors = [
        output for cell in nb.cells if "outputs" in cell
        for output in cell["outputs"] if output.output_type == "error"
    ]

    os.remove(fname)

    return nb, errors


def test_tutorial(capsys):
    fname = os.path.join(tut_path, 'tutorial.ipynb')
    nb, errors = _notebook_run(fname, capsys=capsys)
    assert errors == []
