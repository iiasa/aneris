#!/usr/bin/env python
from __future__ import print_function

import glob
import os
import shutil
import versioneer

from setuptools import setup, Command
from subprocess import call


# Thanks to http://patorjk.com/software/taag/
logo = r"""
     ___      .__   __.  _______ .______       __       _______.
    /   \     |  \ |  | |   ____||   _  \     |  |     /       |
   /  ^  \    |   \|  | |  |__   |  |_)  |    |  |    |   (----`
  /  /_\  \   |  . `  | |   __|  |      /     |  |     \   \    
 /  _____  \  |  |\   | |  |____ |  |\  \----.|  | .----)   |   
/__/     \__\ |__| \__| |_______|| _| `._____||__| |_______/    
"""

REQUIREMENTS = [
    'argparse',
    'numpy',
    # pin to <=1.0 is due to 1.1.0 regression in
    # https://github.com/pandas-dev/pandas/issues/35753
    'pandas>0.24<=1.0',
    'PyYAML',
    'xlrd',
    'xlsxwriter',
    'matplotlib',
    'pyomo>=5'
]

EXTRA_REQUIREMENTS = {
    'tests': ['pytest', 'coverage', 'coveralls', 'pytest', 'pytest-cov'],
    'deploy': ['twine', 'setuptools', 'wheel'],
}


# thank you https://stormpath.com/blog/building-simple-cli-interfaces-in-python
class RunTests(Command):
    """Run all tests."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(['py.test', '--cov=skele', '--cov-report=term-missing'])
        raise SystemExit(errno)


CMDCLASS = versioneer.get_cmdclass()
CMDCLASS.update({'test': RunTests})


def main():
    print(logo)
    classifiers = [
        'License :: OSI Approved :: Apache Software License',
    ]
    packages = [
        'aneris',
    ]
    pack_dir = {
        'aneris': 'aneris',
    }
    entry_points = {
        'console_scripts': [
            # list CLIs here
            'aneris=aneris.cli:main',
        ],
    }
    package_data = {
        # add explicit data files here
        # 'aneris': [],
    }
    install_requirements = REQUIREMENTS
    extra_requirements = EXTRA_REQUIREMENTS
    setup_kwargs = {
        "name": "aneris-iamc",
        'version': versioneer.get_version(),
        "description": 'Harmonize Integrated Assessment Model Emissions '
        'Trajectories',
        "author": 'Matthew Gidden',
        "author_email": 'matthew.gidden@gmail.com',
        "url": 'http://github.com/iiasa/aneris',
        'cmdclass': CMDCLASS,
        'classifiers': classifiers,
        'license': 'Apache License 2.0',
        'packages': packages,
        'package_dir': pack_dir,
        'entry_points': entry_points,
        'package_data': package_data,
        'python_requires': '>=3.6',
        'install_requires': install_requirements,
        'extras_require': extra_requirements,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
