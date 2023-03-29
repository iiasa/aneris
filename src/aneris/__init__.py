from importlib.metadata import version as _version

from aneris._io import *  # noqa: F401, F403
from aneris.cmip6 import cmip6_utils, driver  # noqa: F401, F403
from aneris.harmonize import *  # noqa: F401, F403
from aneris.utils import *  # noqa: F401, F403


try:
    __version__ = _version("aneris")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
