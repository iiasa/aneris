#!/usr/bin/env python
from __future__ import print_function

import glob
import os

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
    'version': '0.1.0',
}


def main():
    print(logo)

    packages = [
        'aneris',
    ]
    pack_dir = {
        'aneris': 'aneris',
    }
    entry_points = {
        'console_scripts': [
            'aneris=aneris.cli:main',
        ],
    }
    setup_kwargs = {
        "name": "aneris",
        "version": INFO['version'],
        "description": 'Harmonize Integrated Assessment Model Emissions '
        'Trajectories',
        "author": 'Matthew Gidden',
        "author_email": 'matthew.gidden@gmail.com',
        "url": 'http://github.com/gidden/aneris',
        "packages": packages,
        "package_dir": pack_dir,
        "entry_points": entry_points,
        "zip_safe": False,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
