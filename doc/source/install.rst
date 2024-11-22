.. _install:

Install
*******

Via Conda (installs dependencies for you)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    conda install -c conda-forge aneris

Via Pip
~~~~~~~

.. code-block:: bash

    pip install aneris-iamc

From Source
~~~~~~~~~~~

.. code-block:: bash

    pip install git+https://github.com/iiasa/aneris.git

Dependencies
~~~~~~~~~~~~

The dependencies for :code:`aneris` are:

  .. program-output:: python -c 'import sys, os; sys.path.append("../.."); import setup; print("\n".join([r for r in setup.REQUIREMENTS]))'
