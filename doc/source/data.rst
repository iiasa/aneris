.. _data:

Data Model
**********

.. todo::

  Update once new data model is agreed to

Data fed into :code:`aneris` is assumed to be so-called wide-style panel
data. Within the code base, standard Pandas :code:`DataFrame`s are used. When
using the CLI, Excel spreadsheets or csvs are supported.

Variable Names
~~~~~~~~~~~~~~

The most important data underlying harmonizations are timeseries. Timeseries are
defined by a Variable and a number of Data columns for each timestep in the
timeseries. Variable names are assumed to follow the IAMC style and have the
form:

.. code-block:: bash

    <prefix>|Emissions|<gas>|<sector>|<suffix>

- **prefix**: a designation for the current study (e.g., "CEDS")
- **gas**: the emissions species (e.g., "BC")
- **sector**: the emissions sector (e.g., "Transportation")
- **suffix**: a designation for raw-model data (e.g., "Unharmonized")

Importantly, **model data variable names must match historical data variable
names exactly**.


Unharmonized IAM Data
~~~~~~~~~~~~~~~~~~~~~

Data from IAMs is expected to be in the following format with a sheetname "data".

.. exceltable:: Example Model Input
   :file: ../../tests/test_data/model_regions_sectors.xls
   :header: 1
   :selection: A1:I4

If overrides are provided, they are expected to be in the following formay with
a sheetname "harmonization".

.. exceltable:: Example Harmonization Overrides
   :file: ../../tests/test_data/model_regions_sectors.xls
   :sheet: 1

Additionally, configuration parameters (described in :ref:`config`) can be set
by two columns titled "Configuration" and "Value" in the harmonization sheet.


Historical Data
~~~~~~~~~~~~~~~

Historical data is expected to be in the following format

.. exceltable:: Example Historical Data
   :file: ../../tests/test_data/history_regions_sectors.xls
   :header: 1
   :selection: A1:I4

Regional Definitions
~~~~~~~~~~~~~~~~~~~~

Data for regional mappings (countries to IAM regions) is expected to be in the
following format

.. csv-table:: Example Regional Definitions
   :file: ../../tests/test_data/regions_regions_sectors.csv
   :header-rows: 1
