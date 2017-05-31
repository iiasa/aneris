#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import subprocess

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

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
    'version': '0.0.1',
}


def main():
    print(logo)

    here = os.path.dirname(os.path.abspath(__file__))
    scripts = glob.glob(os.path.join(here, 'scripts', '*'))

    packages = [
        'aneris',
    ]
    pack_dir = {
        'aneris': 'aneris',
    }
    setup_kwargs = {
        "name": "aneris",
        "version": INFO['version'],
        # update the following:
        "description": 'Harmonize Integrated Assessment Model Emissions '
        'Trajectories',
        "author": 'Matthew Gidden',
        "author_email": 'matthew.gidden@gmail.com',
        "url": 'http://github.com/gidden/aneris',
        "packages": packages,
        "package_dir": pack_dir,
        "scripts": scripts,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
