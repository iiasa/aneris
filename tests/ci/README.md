# Regression Tests on CI

These tests rerun scenarios harmonized with previous versions of `aneris` and
confirm similar results.

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

1. `download_data.sh`
2. `make_output.py`
3. `upload_output.sh`

## `upload_output.sh`

This command must be run by passing your ssh information, so either

```
./upload_db.sh <user>@data.ene.iiasa.ac.at
```

or

```
./upload_db.sh <ssh alias>
```
