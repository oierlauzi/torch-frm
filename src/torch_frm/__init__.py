from importlib.metadata import PackageNotFoundError, version

__author__ = "Oier Lauzirika Zarrabeitia"
__email__ = "olauzirika@cnb.csic.es"

from torch_frm.spherical_harmonics import (
    spherical_harmonics, 
    ravel_spherical_harmonic_index
)

try:
    __version__ = version("torch-frm")
except PackageNotFoundError:
    __version__ = "uninstalled"
    
__all__ = [
    spherical_harmonics,
    ravel_spherical_harmonic_index
]