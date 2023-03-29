import os


try:
    from urllib.request import urlretrieve  # py3
except ImportError:
    from urllib import urlretrieve  # py2

import aneris


_default_cache_dir = os.path.join("~", ".aneris_tutorial_data")


# idea borrowed from Seaborn
def load_data(
    cache_dir=_default_cache_dir,
    cache=True,
    github_url="https://github.com/iiasa/aneris",
):
    """
    Load a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    cache_dir : string, optional
        The directory in which to search for and write cached data.
    cache : boolean, optional
        If True, then cache data locally for use on subsequent calls
    github_url : string
        Github repository where the data is stored
    """
    longdir = os.path.expanduser(cache_dir)
    if not os.path.isdir(longdir):
        os.mkdir(longdir)

    files = {
        "rc": "aneris_regions_sectors.yaml",
        "hist": "history_regions_sectors.xls",
        "model": "model_regions_sectors.xls",
        "regions": "regions_regions_sectors.csv",
    }
    files = {k: os.path.join(longdir, f) for k, f in files.items()}

    for localfile in files.values():
        if not os.path.exists(localfile):
            fname = os.path.basename(localfile)
            url = "/".join((github_url, "raw", "main", "tests", "test_data", fname))
            urlretrieve(url, localfile)

    # read input
    hist = aneris.pd_read(files["hist"])
    if hist.empty:
        raise ValueError("History file is empty")
    hist.columns = hist.columns.astype(str)  # make sure they're all strings
    regions = aneris.pd_read(files["regions"])
    if regions.empty:
        raise ValueError("Region definition is empty")
    model, overrides, config = aneris.read_excel(files["model"])
    # TODO: this is a change to the file-based API, and we should probably
    # develop a deprecation schedule or force updates to downloaded files
    # to address it
    if "Unit" in overrides:
        overrides = overrides.drop(columns=["Unit"])
    model.columns = model.columns.astype(str)  # make sure they're all strings
    rc = aneris.RunControl(rc=files["rc"])
    rc.recursive_update("config", config)

    # get driver
    driver = aneris.cmip6.driver.HarmonizationDriver(
        rc, hist, model, overrides, regions
    )

    if not cache:
        for localfile in files.values():
            os.remove(localfile)

    return model, hist, driver


if __name__ == "__main__":
    model, hist, driver = load_data(cache=False)
    for scenario in driver.scenarios():
        driver.harmonize(scenario)
    harmonized, metadata, diagnostics = driver.harmonized_results()
