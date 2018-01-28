#!/usr/bin/env python
from __future__ import print_function

import glob
import os
import shutil

from setuptools import setup, Command, find_packages
from setuptools.command.install import install


# Thanks to http://patorjk.com/software/taag/
logo = r"""
     ___      .__   __.  _______ .______       __       _______.
    /   \     |  \ |  | |   ____||   _  \     |  |     /       |
   /  ^  \    |   \|  | |  |__   |  |_)  |    |  |    |   (----`
  /  /_\  \   |  . `  | |   __|  |      /     |  |     \   \    
 /  _____  \  |  |\   | |  |____ |  |\  \----.|  | .----)   |   
/__/     \__\ |__| \__| |_______|| _| `._____||__| |_______/    
"""

INFO = {
    'version': '0.1.0',
}


class Cmd(install):
    """Custom clean command to tidy up the project root."""

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        install.run(self)
        dirs = [
            'aneris_iamc.egg-info'
        ]
        for d in dirs:
            print('removing {}'.format(d))
            shutil.rmtree(d)


def main():
    print(logo)

    packages = find_packages()
    pack_dir = {
        'aneris': 'aneris',
    }
    entry_points = {
        'console_scripts': [
            'aneris=aneris.cli:main',
        ],
    }
    cmdclass = {
        'install': Cmd,
    }
    setup_kwargs = {
        "name": "aneris-iamc",
        "version": INFO['version'],
        "description": 'Harmonize Integrated Assessment Model Emissions '
        'Trajectories',
        "author": 'Matthew Gidden',
        "author_email": 'matthew.gidden@gmail.com',
        "url": 'http://github.com/gidden/aneris',
        "packages": packages,
        "package_dir": pack_dir,
        "entry_points": entry_points,
        "cmdclass": cmdclass,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
