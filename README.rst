`aneris` - Harmonization for IAMs
=================================
   
**Please note that aneris is still in early developmental stages, thus all interfaces are subject to change.**

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
