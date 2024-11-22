`aneris` - Operations library for harmonizing, downscaling, and gridding IAM emissions data
===========================================================================================

**To reproduce harmonization routines from [Gidden et al.
(2019)](https://gmd.copernicus.org/articles/12/1443/2019/), use `v0.3.2` (or
earlier). Subsequent versions introduce backwards incompatibilities.**

Documentation
-------------

All documentation can be found at https://aneris.readthedocs.io/en/latest/

Install
-------

From Source
***********

Installing from source is as easy as

.. code-block:: bash

    pip install -e .[tests,deploy,units]

You can then check to make sure your install is operating as expected

.. code-block:: bash

    pytest tests

Build the Docs
--------------

Requirements
************

See `doc/environment.yml`

Build and Serve
***************

.. code-block:: bash

    cd doc
    make html

Then point you browser to `http://127.0.0.1:8000/`.

License
-------

Licensed under Apache 2.0. See the LICENSE file for more information
