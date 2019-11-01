# Regression Tests on CI

These tests rerun scenarios harmonized with previous versions of `aneris` and
confirm similar results.

# TODO

1. The listing of datasets should be housed in one location. At the moment, it is explicitly listed in three.
2. After that, the tests in `test_regression.py` should be parameterized on that list

# Test Data Location

This folder is used for generating a database of testable scenario
instances. You must have the following envrionment variables set to execute all
aspects:

- `ANERIS_CI_USER`
- `ANERIS_CI_PW`


The values for these variables can be found in the internal IIASA document
[here](https://iiasahub.sharepoint.com/:w:/s/ene/EaDqCDLy9AVLuY7YUm3DctcBJzBiLXv6ffecxdUgW5oZ4A).

# Generate the Database

To generate and upload the database execute the follow files in order:

1. `download_data.py`
2. `make_output.sh`
3. `upload_output.sh`

## `upload_output.sh`

This command must be run by passing your ssh information, so either

```
./upload_output.sh <user>@data.ene.iiasa.ac.at
```

or

```
./upload_output.sh <ssh alias>
```

# Change Input Data

If input data needs to be changed you can do so by:


1. `download_data.py`
2. Do adjustments to input data as needed
3. `upload_input.sh`

## `upload_input.sh`

This command must be run by passing your ssh information, so either

```
./upload_input.sh <user>@data.ene.iiasa.ac.at
```

or

```
./upload_input.sh <ssh alias>
```
