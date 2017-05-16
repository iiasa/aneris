A Python Project Stub
=====================

This is a stub for a new python project. It contains the following features:

1. Installable module
2. Unit tests
3. Continuous integration on Linux and Windows
4. Automatic documentation

Dependencies
------------

This stub depends on

1. `nose <https://pypi.python.org/pypi/nose/1.3.7>`_
2. `Sphinx <https://pypi.python.org/pypi/Sphinx>`_

Making a New Project
--------------------

Choose a project name and run the following commands replacing ``<project
name>`` with your choice::

    git clone https://github.com/gidden/aneris.git <project name>
    cd <project name>
    ./make_proj.sh <project name>

You should now have a fresh new repository with your project ready to go. You
can sync it with Github via::

    git remote add origin git@github.com:<user name>/<project name>
    git push origin master

Installing
----------

You can install the stub like any other python module::

    ./setup.py install

or for local installations::

    ./setup.py install --user

Testing
----------

From the root directory, run::

    nosetests -w tests

Continuous Integration
-----------------------

Once you have added your project to Github (or any other supported service), you
can turn on continuous integration. Once turned on, future pull requests will be
automatically tested.

Linux builds are tested on TravisCI. You can turn on the new project at
``https://travis-ci.com/profile/<user name>`` if the repository is *private* or
``https://travis-ci.org/profile/<user name>`` if the repository is
*public*. 

Windows builds are tested on `Appveyor <https://ci.appveyor.com/projects>`_. You
can add it by clicking the "New Project" button.

Documentation
--------------

On ReadTheDocs
~~~~~~~~~~~~~~~

This repository currently has its documentation on `ReadTheDocs
<http://aneris.readthedocs.org/en/latest/>`_. You can add any new project and
have it automatically hosted by doing the following

- on Github

  - Setting -> Webhooks and Services-> Add Service -> ReadTheDocs

- on ReadTheDocs

  - Add the project
  - On readthedocs.org/projects/<project name>, do Admin -> Advanced Settings ->
    Check the "Install Project" Box at the top

On *Nix Platforms
~~~~~~~~~~~~~~~~~

After you install the project locally, you can generate documentation by::

    cd docs
    make html

You can serve the documentation locally via::

    make serve
	
You can then view the docs at http://localhost:8000/build/html/

On Windows
~~~~~~~~~~~~~~~~~

Follow the above instructions replacing ``make`` with ``./make.bat``.
