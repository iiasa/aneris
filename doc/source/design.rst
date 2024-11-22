.. design_:

Design
******

`aneris` is designed to connect a series of modular components that enable the
processing of IAM scenario data into climate-model-ready data. The following
modules are currently supported:

- `Harmonization`: aligns IAM model data with historical data
- `Downscaling`: provides higher-resolution data from IAM data
- `Gridding`: provides grid-level data from high-resolution IAM data
- `Climate`: uses IAM emission data to generate concentration timeseries using
  an `openscm`-aware climate model

All `Module` interfaces are designed to be inheritable to support third-party
implementations. A top-level `Workflow` can be configured to create process
workflows that utilize either the standard `Module`s provided with the `aneris`
package, or third-party variants if they are installed.

Harmonization
~~~~~~~~~~~~~

The `Harmonization` module takes as input

1. IAM model data at a given region and variable (sector and gas by default)
   resolution
2. Historical data at a given region and variable (sector and gas by default)
   resolution
3. Decision logic to decide which method to use for harmonization (see below
   examples) (optional)

It then harmonizes the IAM data to historical data based either on default logic
or via user-provided logic.

It provides as output

1. Harmonized data at the same region and variable (sector and gas by default)
   resolution

The module is described in more detail in the following sections

.. toctree::
   :maxdepth: 2

   tutorial.ipynb
   theory
   budget_method.ipynb
   config

.. todo::

  Add documentation for logic

Downscaling
~~~~~~~~~~~

The `Downscaling` module implements different downscaling routines to enhance
the spatial resolution of data. It reqiures

1. IAM model data at a given region and variable (sector and gas by default)
   resolution - in a standard workflow, this would be the output of the
   `Harmonization` module
2. Historical data at a given variable (sector and gas by default) resolution
   and **higher spatial resolution** (e.g., at the country-level)
3. Scenario data (e.g., population, GDP - depending on the downscaling method
   used) at **higher spatial resolution** (the same as the historical data)
4. Decision logic mapping each sector or gas to a downscaling method (optional)

.. warning::

  Historical data is assumed to be consistent with model data in the first model
  time period.

It provides as output

1. IAM data at a given variable (sector and gas by default) resolution and at
   the *higher spatial resolution* of the historical data used

.. todo::

  Add documentation for logic

Gridding
~~~~~~~~

The `Gridding` module generates spatial grids of emissions data compliant with
CMIP/ESGF dataformats

It takes as input

1. IAM data at the *country-level* defined by emissions species and sector -
   normally an output of the `Downscaling` module
2. Gridded data of proxy patterns
3. A mapping of which proxy patterns to use for which emissions species and sector

It provides as output

1. Gridded IAM data at the provided emissions species and sector levels with
   ESGF-compliant metadata

.. todo::

  Add documentation for installing pattern files

Climate
~~~~~~~

.. todo::

  Develop in tandem with openscm developers

Workflow
~~~~~~~~

.. todo::

  Write documentation once we have some example workflows
